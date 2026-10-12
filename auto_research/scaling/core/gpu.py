"""User-authorized, preemptible GPU reservations owned by the research queue."""

from __future__ import annotations

import os
import signal
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .common import CODE, alive, config, identity, launch, log, read, root, stop, write


def folder():
    return root() / "state/gpu_reservations"


def settings():
    path = root() / "config/operations.json"
    return read(path).get("gpu_reservation", {}) if path.exists() else {}


def save(records):
    write(folder() / "processes.json", {"time": time.time(), "gpus": records})


def load():
    from .queue import adopt

    path = folder() / "processes.json"
    records = read(path)["gpus"] if path.exists() else {}
    for key, record in list(records.items()):
        adopt(record)
        if not alive(record.get("identity")) and (
            record["status"] != "dispatching" or time.time() - record["started_at"] > 60
        ):
            log("gpu_reservation_exited", gpu=int(key))
            records.pop(key)
    save(records)
    return records


def external_gpus(inventory, records):
    """Never preempt external processes, including processes sharing a held GPU."""
    groups = {
        rec["identity"]["pid"] for rec in records.values() if alive(rec.get("identity"))
    }
    uuids = {row["uuid"]: row["gpu"] for row in inventory}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    external = set()
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or fields[0] not in uuids:
            continue
        try:
            owned = os.getpgid(int(fields[1])) in groups
        except (OSError, ValueError):
            owned = False
        if not owned:
            external.add(uuids[fields[0]])
    return external


def available(inventory, records, leased, external):
    cfg = config()
    return [
        row["gpu"]
        for row in inventory
        if row["gpu"] in cfg["gpus"]
        and row["gpu"] not in leased | external
        and row["free_mib"] + (3072 if str(row["gpu"]) in records else 0)
        >= cfg["free_memory_mib"]
        and (
            str(row["gpu"]) in records
            or row["utilization"] <= cfg["idle_utilization_percent"]
        )
    ]


def release(records, devices):
    for device in devices:
        record = records.get(str(device))
        if record is None:
            continue
        state = Path(record["intent"]).parent
        write(state / "cancel.json", {"time": time.time()})
        witness = record.get("identity")
        path = state / "identity.json"
        if witness is None and path.exists():
            witness = read(path)
        stop(witness)
        if alive(witness):
            raise RuntimeError("GPU reservation did not release")
        records.pop(str(device))
        save(records)
        log("gpu_reservation_released", gpu=device)


def stop_all():
    records = load()
    release(records, [int(key) for key in records])


def snapshot():
    path = folder() / "processes.json"
    records = read(path)["gpus"] if path.exists() else {}
    return [
        {"gpu": int(key), "status": rec["status"], "alive": alive(rec.get("identity"))}
        for key, rec in records.items()
    ]


def maintain(records, devices):
    options = settings()
    if not options.get("enabled"):
        release(records, [int(key) for key in records])
        return
    for device in devices:
        if str(device) in records:
            continue
        directory = folder() / str(device)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "ready.json").unlink(missing_ok=True)
        # Never reuse a cancelled handshake: a delayed wrapper must see its own
        # cancellation, even if the next reservation has already started.
        for previous in directory.glob("launch-*"):
            if time.time() - previous.stat().st_mtime > 900:
                witness = previous / "identity.json"
                if not witness.exists() or not alive(read(witness)):
                    shutil.rmtree(previous)
        state = directory / ("launch-" + uuid.uuid4().hex)
        state.mkdir()
        intent = state / "intent.json"
        write(
            intent,
            {
                "command": [
                    config()["training_python"],
                    str(CODE / "pipeline.py"),
                    "worker",
                    "gpu-reserve",
                    "--gpu",
                    str(device),
                ],
                "gpus": [device],
                "job": "gpu-reservation",
                "time": time.time(),
            },
        )
        records[str(device)] = {
            "status": "dispatching",
            "identity": None,
            "intent": str(intent),
            "started_at": time.time(),
        }
        save(records)
        process = launch(
            [
                config()["training_python"],
                str(CODE / "pipeline.py"),
                "execute",
                "--intent",
                str(intent),
            ],
            root() / "logs/reservations" / (str(device) + ".log"),
            [device],
        )
        records[str(device)].update(status="running", identity=identity(process.pid))
        save(records)
        write(state / "ack.json", {"time": time.time()})
        log("gpu_reservation_started", gpu=device, pid=process.pid)


def hold(device):
    """Hold only an already-allocated GPU; a bounded duty cycle permits fast handoff."""
    import torch

    options = settings()
    if not options.get("enabled") or device not in config()["gpus"]:
        return
    finished = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: finished.set())
    signal.signal(signal.SIGINT, lambda *_: finished.set())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    memory = torch.empty(1024 * 2**20, dtype=torch.uint8, device="cuda")
    memory.zero_()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    output = torch.empty_like(a)
    write(
        folder() / str(device) / "ready.json",
        {
            "time": time.time(),
            "gpu": device,
            "worker": identity(os.getpid()),
            "kind": "resource_reservation",
            "research_optimizer_steps": 0,
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        },
    )
    lease = folder() / "processes.json"
    while not finished.is_set():
        if (
            any(
                (root() / "state" / name).exists()
                for name in ("STOP", "complete.json", "active_complete.json")
            )
            or not lease.exists()
            or time.time() - lease.stat().st_mtime > 900
            or not settings().get("enabled")
        ):
            break
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and not finished.is_set():
            torch.mm(a, b, out=output)
            torch.cuda.synchronize()
        finished.wait(0.75)
