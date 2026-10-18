"""Frozen evaluator snapshots; serving GPUs released before judge-only work."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .common import (
    BASE,
    BENCHMARKS,
    PROJECT,
    DataError,
    Waiting,
    alive,
    config,
    credentials_ready,
    env,
    identity,
    read,
    root,
    source_module,
    sha,
    stop,
    write,
)
from .train import directory, run_record


def evaluation_env():
    result = env()
    extra = config()["evaluation_dependencies"]
    result["PYTHONPATH"] = extra + (
        ":" + result["PYTHONPATH"] if result.get("PYTHONPATH") else ""
    )
    result["NLTK_DATA"] = str(Path(extra).parent / "nltk_data")
    return result


def frozen_sources():
    paths = []
    for model in config()["models"]:
        paths.extend(
            BASE / "NEU" / model / "evaluation" / name
            for name in (
                "evaluate.py",
                "verify_results.py",
                "config.json",
            )
        )
        paths.append(
            PROJECT / "test_scripts" / model / "evaluation_sources.sha256.json"
        )
        model_path = Path(config()["models"][model])
        paths.extend(
            p
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "generation_config.json",
            )
            if (p := model_path / name).is_file()
        )
    paths.extend(
        PROJECT / "training_data/sample" / name
        for name in ("dev_indices.jsonl", "used_train_indices.jsonl", "manifest.json")
    )
    paths.extend(
        BASE / "configs/prompts" / (m + ".txt")
        for m in ("NEU", "SUP", "AUG-SUP", "SSR")
    )
    paths.append(BASE / "configs/experiment.json")
    paths.extend((BASE / "core").glob("*.py"))
    # Freeze datasets as well as scripts; interpreter bytecode is generated at import.
    paths.extend(
        p
        for p in (PROJECT / "test_data").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    return {str(p): sha(p) for p in paths}


def check_sources():
    frozen = read(root() / "config/frozen_sources.json")
    for name, expected in frozen.items():
        if sha(Path(name)) != expected:
            raise DataError("A frozen input/evaluator changed: " + name)


def evaluation_directory(run_id, milestone):
    return directory(run_id, milestone) / "evaluation"


def provision(run_id, milestone):
    check_sources()
    record = run_record(run_id)
    out = evaluation_directory(run_id, milestone)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime").mkdir(exist_ok=True)
    source = BASE / "NEU" / record["model"] / "evaluation"
    for name in ("evaluate.py", "verify_results.py"):
        target = out / name
        if target.exists() and sha(target) != sha(source / name):
            raise DataError("Frozen evaluator snapshot changed")
        if not target.exists():
            shutil.copy2(source / name, target)
    sources = (
        PROJECT / "test_scripts" / record["model"] / "evaluation_sources.sha256.json"
    )
    target_sources = out / sources.name
    if target_sources.exists() and sha(target_sources) != sha(sources):
        raise DataError("Frozen upstream evaluator manifest changed")
    if not target_sources.exists():
        shutil.copy2(sources, target_sources)
    original = read(source / "config.json")
    index = next(
        i
        for i, r in enumerate(read(root() / "manifest.json")["runs"])
        if r["run_id"] == run_id
    )
    ports = [8400 + index * 2, 8401 + index * 2]
    if record["model"] == "Qwen_38_27B":
        ports = ports[:1]
    target = {
        **original,
        "target": {
            **original["target"],
            "model": "scale-" + run_id.lower() + "-" + str(milestone),
            "base_urls": [f"http://127.0.0.1:{p}/v1" for p in ports],
        },
    }
    path = out / "config.json"
    if path.exists() and read(path) != target:
        raise DataError("Frozen evaluation configuration changed")
    if not path.exists():
        write(path, target)
    for bench in BENCHMARKS:
        (out / bench).mkdir(exist_ok=True)
    return out, target, ports


def cleanup_services(run_id, milestone):
    path = evaluation_directory(run_id, milestone) / "runtime/service_state.json"
    if path.exists():
        state = read(path)
        for process in state["servers"]:
            stop(process["identity"])
        write(
            path.with_name("service_history.json"), {**state, "stopped_at": time.time()}
        )
        path.unlink()


def child(out, bench, stage):
    log = out / "runtime" / (bench + "_" + stage + ".log")
    with log.open("a") as stream:
        subprocess.run(
            [
                config()["evaluation_python"],
                str(Path(__file__).resolve().parents[1] / "pipeline.py"),
                "worker",
                "frozen-eval",
                "--evaluation-dir",
                str(out),
                "--bench",
                bench,
                "--stage",
                stage,
            ],
            cwd=out,
            env=evaluation_env(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=config()["inference_timeout_hours"] * 3600,
        )


def frozen_child(out, bench, stage):
    """Run the unchanged evaluator with stricter shared transport admission only."""
    from .api import remote_request_slot, reserve_quota

    module = source_module(Path(out) / "evaluate.py", "frozen_scoped_evaluator")
    module.reserve_quota = reserve_quota
    original = module.request

    def request(spec, *args, **kwargs):
        if spec.get("local"):
            return original(spec, *args, **kwargs)
        with remote_request_slot():
            return original(spec, *args, **kwargs)

    module.request = request
    sys.argv = [str(Path(out) / "evaluate.py"), "--bench", bench, "--stage", stage]
    module.main()


def generate(run_id, milestone):
    cfg = config()
    out, evaluation, ports = provision(run_id, milestone)
    record = run_record(run_id)
    trained = read(directory(run_id, milestone) / "training/complete.json")
    model = Path(trained["merged_model"])
    index = read(model / "model.safetensors.index.json")
    if not index["weight_map"] or any(
        not (model / p).is_file() for p in set(index["weight_map"].values())
    ):
        raise DataError("Merged trained model is incomplete")
    physical = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    groups = (
        [physical] if record["model"] == "Qwen_38_27B" else [[gpu] for gpu in physical]
    )
    if len(groups) != len(ports):
        raise DataError("Serving GPU assignment differs from frozen replica plan")
    cleanup_services(run_id, milestone)
    state = {
        "run_id": run_id,
        "milestone": milestone,
        "started_at": time.time(),
        "servers": [],
    }
    try:
        for devices, port in zip(groups, ports):
            with socket.socket() as test:
                try:
                    test.bind(("127.0.0.1", port))
                except OSError:
                    raise Waiting("Reserved inference port is occupied") from None
            name = evaluation["target"]["model"]
            environment = env(devices)
            environment.update(
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                TRITON_CACHE_DIR=f"/home/tiger/.cache/{name}-{port}/triton",
                TORCHINDUCTOR_CACHE_DIR=f"/home/tiger/.cache/{name}-{port}/torchinductor",
            )
            kwargs = {"enable_thinking": True}
            if record["model"] == "Qwen_38_27B":
                kwargs["reasoning_effort"] = "medium"
            command = [
                cfg["vllm_python"],
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--served-model-name",
                name,
                "--dtype",
                "bfloat16",
                "--tensor-parallel-size",
                str(len(devices)),
                "--max-model-len",
                "65536",
                "--max-num-seqs",
                "32",
                "--max-num-batched-tokens",
                "16384",
                "--gpu-memory-utilization",
                "0.88",
                "--enable-prefix-caching",
                "--enable-chunked-prefill",
                "--gdn-prefill-backend",
                "triton",
                "--language-model-only",
                "--reasoning-parser",
                "qwen3",
                "--default-chat-template-kwargs",
                json.dumps(kwargs),
                "--compilation-config",
                '{"mode":3}',
            ]
            with (out / "runtime" / ("server-" + str(port) + ".log")).open("a") as log:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    cwd=out,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            state["servers"].append(
                {"identity": identity(process.pid), "gpus": devices, "port": port}
            )
            write(out / "runtime/service_state.json", state)
        deadline = time.monotonic() + 1800
        while True:
            ready = True
            for service in state["servers"]:
                if not alive(service["identity"]):
                    raise RuntimeError("vLLM exited before readiness")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{service['port']}/v1/models", timeout=3
                    ) as response:
                        models = json.load(response)["data"]
                    ready &= evaluation["target"]["model"] in {m["id"] for m in models}
                except (OSError, ValueError, KeyError):
                    ready = False
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("vLLM readiness timed out")
            time.sleep(10)
        for bench in BENCHMARKS:
            child(out, bench, "generate")
            write(
                out / "runtime" / ("generated_" + bench + ".json"),
                {
                    "complete": True,
                    "responses_sha256": sha(out / bench / "responses.jsonl"),
                    "time": time.time(),
                },
            )
    finally:
        cleanup_services(run_id, milestone)


def score(run_id, milestone, bench):
    if bench != "ifeval" and not credentials_ready():
        raise Waiting("Judge credentials unavailable")
    out, _, _ = provision(run_id, milestone)
    child(out, bench, "score")
    result = read(out / bench / "metrics.json")
    if result.get("status") != "complete":
        raise DataError("Evaluator did not produce a complete score")


def verify(run_id, milestone):
    out, _, _ = provision(run_id, milestone)
    with (out / "runtime/verify.log").open("a") as stream:
        subprocess.run(
            [config()["evaluation_python"], str(out / "verify_results.py")],
            cwd=out,
            env=evaluation_env(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def preflight():
    """Re-score the existing 541 base answers to detect dependency-induced drift."""
    script = """import importlib.util,json,sys,os,importlib.metadata,nltk,socket
from pathlib import Path
socket.setdefaulttimeout(30)
try:nltk.data.find("tokenizers/punkt_tab/english/")
except LookupError:nltk.download("punkt_tab",download_dir=os.environ["NLTK_DATA"],quiet=True,raise_on_error=True)
base=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("frozen_eval",base/"evaluate.py");E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
actual=E.score_ifeval(E.questions("ifeval"),E.load_lines(base/"ifeval/responses.jsonl"),out)
expected=json.loads((base/"ifeval/metrics.json").read_text())["metrics"]
old=E.load_lines(base/"ifeval/judgments.jsonl");new=E.load_lines(out/"judgments.jsonl")
assert set(old)==set(new),"IFEval denominator changed"
drift=[]
for rid,a in old.items():
 b=new[rid];assert a["instruction_ids"]==b["instruction_ids"]
 for mode in ["strict","loose"]:
  assert len(a[mode])==len(b[mode])
  for i,(left,right) in enumerate(zip(a[mode],b[mode])):
   if left!=right:
    assert rid in ["1122","1129"] and a["instruction_ids"][i]=="keywords:letter_frequency","Unexpected deterministic checker drift"
    drift.append({"id":rid,"mode":mode,"instruction":"keywords:letter_frequency","old":left,"new":right})
E.eq_helpers()
report={"passed":True,"ifeval_existing_answers":541,"exact_metrics_match":actual==expected,"deterministic_instructions_match":True,"known_stochastic_checker_drift":drift,"old_metrics":expected,"new_metrics":actual,"versions":{n:importlib.metadata.version(n) for n in ["nltk","langdetect","numpy","scipy","wordfreq"]}}
(out/"complete.json").write_text(json.dumps(report,indent=2)+"\\n")
"""
    out = root() / "state/evaluator_preflight"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "check.log").open("a") as log:
        subprocess.run(
            [
                config()["evaluation_python"],
                "-c",
                script,
                str(PROJECT / "test_scripts/Qwen_35_9B"),
                str(out),
            ],
            env=evaluation_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
