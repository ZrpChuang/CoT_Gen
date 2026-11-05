#!/usr/bin/env python3
"""Prepare audited CoT artifacts, publish privately to HF, and verify the upload.

prepare uses only Python's standard library. publish requires huggingface_hub
and safetensors, installed in an isolated environment. Original files are never
deleted. An inventory describes the exact allowlist; directory globs are not
used to upload experiment roots.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tarfile
import tempfile
import io


SECRETS = [
    re.compile(rb"hf_[A-Za-z0-9]{25,192}"),
    re.compile(rb"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{24,128}_GPT_AK"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{32,256}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"(?i)((?:api[_-]?key|access[_-]?token|password)\s*[\"']?\s*[:=]\s*[\"'])([A-Za-z0-9_-]{24,512})([\"'])"),
]
UNSAFE_NAMES = {
    "local_paths.json", "deployment.json", "api_credentials.json", ".env",
    "operations.json", "credentials.json", "training_args.bin",
}
CORE = {
    "fresh_generated_dataset", "fresh_generation_database",
    "fresh_evaluation_and_training_evidence", "fresh_resume_extras",
    "historical_evidence", "historical_baseline_dataset_and_evaluation",
    "historical_pattern_archive", "historical_pattern_dataset_and_evaluation",
    "base_evaluation_evidence", "review_metadata",
}
SIDECARS = {
    "adapter_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "generation_config.json", "vocab.json", "merges.txt",
}
TEXT_TYPES = {".json", ".jsonl", ".md", ".txt", ".tsv", ".yaml", ".yml", ".py", ".sh", ".jinja", ".toml"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unsafe(path):
    p = Path(path)
    return (
        p.name in UNSAFE_NAMES or p.name.startswith(".env")
        or p.suffix in {".pid", ".lock", ".pyc", ".log", ".wandb", ".key", ".pem"}
        or any(x in {".git", "__pycache__"} for x in p.parts)
        or "credential" in p.name.lower()
    )


def scan_stream(stream, label):
    tail = b""
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        data = tail + chunk
        if any(pattern.search(data) for pattern in SECRETS):
            raise ValueError("Possible credential; inspect privately before upload: " + label)
        tail = data[-512:]


def check_file(path, label):
    if path.suffix == ".safetensors":
        # Credentials can only occur in this format's JSON header, not tensors.
        with path.open("rb") as f:
            length = int.from_bytes(f.read(8), "little")
            if not 0 < length < min(path.stat().st_size, 128 * 1024 * 1024):
                raise ValueError("Invalid safetensors header: " + label)
            scan_stream(io.BytesIO(f.read(length)), label)
        return
    with path.open("rb") as f:
        scan_stream(f, label)
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if unsafe(member.name):
                    raise ValueError("Archive contains excluded file: " + label + ":" + member.name)
                with archive.extractfile(member) as stream:
                    scan_stream(stream, label + ":" + member.name)


def redact(data):
    count = 0
    for index, pattern in enumerate(SECRETS):
        replacement = (lambda m: m.group(1) + b"[REDACTED_CREDENTIAL]" + m.group(3)) if index == 4 else b"[REDACTED_CREDENTIAL]"
        data, n = pattern.subn(replacement, data)
        count += n
    return data, count


def sanitized_file(original, temporary, label):
    """Create a sanitized backup copy when needed; never modify the source."""
    destination = Path(temporary) / (hashlib.sha256(label.encode()).hexdigest() + original.suffix)
    redactions, omitted = 0, 0
    if original.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(original, "r:gz") as source, tarfile.open(destination, "w:gz", compresslevel=1) as output:
            for member in source:
                if not member.isfile():
                    continue
                if unsafe(member.name):
                    omitted += 1
                    continue
                if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                    raise ValueError("Unsafe archive path: " + label)
                with source.extractfile(member) as f:
                    payload = f.read()
                if Path(member.name).suffix in TEXT_TYPES:
                    payload, n = redact(payload)
                    redactions += n
                scan_stream(io.BytesIO(payload), label + ":" + member.name)
                info = tarfile.TarInfo(member.name)
                info.mode, info.size = 0o600, len(payload)
                output.addfile(info, io.BytesIO(payload))
        return destination, {"credential_redactions": redactions, "excluded_archive_members": omitted,
                             "archive_repacked": True}
    if original.suffix in TEXT_TYPES:
        with original.open("rb") as source, destination.open("wb") as output:
            for line in source:
                line, n = redact(line)
                redactions += n
                output.write(line)
        if redactions:
            return destination, {"credential_redactions": redactions}
        destination.unlink()
    return original, {}


def sanitize_database(path):
    changes = []
    with sqlite3.connect(path) as connection:
        tables = [r[0] for r in connection.execute("select name from sqlite_master where type='table'")]
        for table in tables:
            if table.startswith("sqlite_"):
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            columns = [r[1] for r in connection.execute("pragma table_info(" + quoted + ")")]
            for row in connection.execute("select rowid,* from " + quoted):
                for index, value in enumerate(row[1:]):
                    if not isinstance(value, (str, bytes)):
                        continue
                    encoded = value.encode() if isinstance(value, str) else value
                    safe, count = redact(encoded)
                    if not count:
                        continue
                    column = '"' + columns[index].replace('"', '""') + '"'
                    replacement = safe.decode() if isinstance(value, str) else safe
                    connection.execute("update " + quoted + " set " + column + "=? where rowid=?", (replacement, row[0]))
                    changes.append({"table": table, "column": columns[index], "redactions": count})
        connection.commit()
        connection.execute("vacuum")  # Remove stale text from free pages as well.
        assert connection.execute("pragma integrity_check").fetchone()[0] == "ok"
        assert connection.execute("select count(*) from accepted").fetchone()[0] == 10000
    return changes


def adapter_destination(path):
    parts = Path(path).parts
    if "gpt_fresh_100k_20261012" in parts:
        method, model = parts[parts.index("runs") + 1].split("__")
        return Path("fresh_10k") / method / model
    if "pattern_20261004" in parts:
        return Path("historical_pattern") / parts[parts.index("models") + 1] / "Qwen_35_9B"
    if "diagnostics" in parts:
        return Path("historical_lr") / "NEU" / "Qwen_35_9B"
    index = parts.index("baseline_method")
    return Path("historical_main") / parts[index + 1] / parts[index + 2]


def package_name(category, path):
    if category.startswith("fresh_"):
        return "fresh_10k_resume" if category == "fresh_resume_extras" else "fresh_10k_data_and_evidence"
    if category.startswith("historical_pattern"):
        return "historical_pattern"
    if category.startswith("historical_"):
        return "historical_baseline"
    if category == "base_evaluation_evidence":
        return "base_evaluation"
    return "review_and_provenance"


def preserve_base_provenance(inventory, models):
    descriptions = []
    for name in sorted({a["base_model_name_or_path"] for a in inventory["final_adapters"]}):
        base = Path(name)
        description = {"local_name": name, "public_revision": None, "files": {},
                       "full_base_weight_sha256_verified": False}
        for filename in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
            path = base / filename
            if path.is_file():
                description["files"][filename] = {"sha256": digest(path), "bytes": path.stat().st_size,
                                                   "resolved_source": str(path.resolve())}
        index = json.loads((base / "model.safetensors.index.json").read_text())
        description["shards"] = []
        for filename in sorted(set(index["weight_map"].values())):
            path = base / filename
            with path.open("rb") as f:
                length_bytes = f.read(8)
                header = f.read(int.from_bytes(length_bytes, "little"))
            description["shards"].append({"filename": filename, "bytes": path.stat().st_size,
                "resolved_source": str(path.resolve()), "header_sha256": hashlib.sha256(length_bytes + header).hexdigest()})
        config = json.loads((base / "config.json").read_text())
        description["model_type"] = config.get("model_type")
        if base.name == "Qwen3.5-9B":
            description["public_model_url"] = "https://huggingface.co/Qwen/Qwen3.5-9B"
        if config.get("model_type") == "qwen3_5_text":
            destination = models / "provenance" / "text_27b"
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ["config.json", "model.safetensors.index.json"]:
                shutil.copy2(base / filename, destination / filename)
            conversion = base.parent / "prepare_text_checkpoint.py"
            if conversion.is_file():
                check_file(conversion, "available text conversion script")
                shutil.copy2(conversion, destination / "prepare_text_checkpoint.py")
            description["conversion_note"] = "Text-only configuration and index preserved. The available conversion script is retained as evidence, not proof of this exact view's creation history."
        descriptions.append(description)
    write_json(models / "base_sources.json", {"models": descriptions,
        "limitation": "Public commit identifiers were absent locally. Header/config fingerprints do not prove full numerical equivalence. Existing merged weights must remain until restoration is verified."})


def prepare(args):
    inventory = json.loads(args.inventory.read_text())
    source = Path(inventory["root"])
    if not source.is_dir():
        raise ValueError("Run prepare on the host mounting the inventoried root")
    target = args.stage.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if (target / "prepared.json").exists():
        raise ValueError("A completed snapshot already exists; use a new directory")
    data = target / "dataset"
    models = target / "model"
    (data / "packages").mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    preserve_base_provenance(inventory, models)
    records, exclusions = [], []
    selected = []
    # Select only inventoried files; configurations of live jobs are excluded.
    for record in inventory["files"]:
        category, rel = record["category"], record["path"]
        if category not in CORE or unsafe(rel):
            continue
        if category == "review_metadata":
            if not ("/reports/" in rel or "/source/" in rel or rel.endswith("/manifest.json")):
                continue
            if "/evidence/paper/" in rel:
                continue
        if "/config/" in rel or "/configs/" in rel or "/runtime/" in rel:
            continue
        # Final adapter payloads go to the model repository, once per run.
        if "/adapter/" in rel or rel.endswith("/adapter_model.safetensors"):
            continue
        selected.append(record)
    # SQLite random writes must use the host's local scratch, not the shared mount.
    with tempfile.TemporaryDirectory(prefix="cot-artifact-snapshot-") as temporary:
        database, database_changes = None, []
        for record in selected:
            if record["category"] != "fresh_generation_database":
                continue
            original = source / record["path"]
            print("SNAPSHOT_SQLITE", flush=True)
            database = Path(temporary) / "generation.sqlite"
            connection = sqlite3.connect(original.as_uri() + "?mode=ro", uri=True)
            with sqlite3.connect(database) as destination:
                connection.backup(destination)
                assert destination.execute("pragma integrity_check").fetchone()[0] == "ok"
                accepted = destination.execute("select count(*) from accepted").fetchone()[0]
                assert accepted == 10000, accepted
            connection.close()
            database_changes = sanitize_database(database)
            print("SQLITE_READY", len(database_changes), "redacted cells", flush=True)
        groups = {}
        for record in selected:
            groups.setdefault(package_name(record["category"], record["path"]), []).append(record)
        for name, group in sorted(groups.items()):
            print("PREPARE", name, len(group), "files", flush=True)
            partial = data / "packages" / (name + ".tar.gz.part")
            final = data / "packages" / (name + ".tar.gz")
            receipt_path = target / "work" / (name + ".json")
            if final.is_file() and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                unchanged = all((source / r["source"]).stat().st_mtime_ns == r["source_mtime_ns"]
                                and (source / r["source"]).stat().st_size == r["source_bytes"]
                                for r in receipt["members"])
                if unchanged and digest(final) == receipt["package_sha256"]:
                    records.extend(receipt["members"])
                    print("REUSE", name, flush=True)
                    continue
            members = []
            with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
                for record in sorted(group, key=lambda r: r["path"]):
                    rel = record["path"]
                    original = source / rel
                    actual = database if record["category"] == "fresh_generation_database" else original
                    if not original.is_file() or original.is_symlink():
                        raise ValueError("Source missing or changed type: " + rel)
                    if original.stat().st_size != record["bytes"]:
                        raise ValueError("Source changed since inventory: " + rel)
                    actual, edits = sanitized_file(actual, temporary, rel)
                    if record["category"] == "fresh_generation_database" and database_changes:
                        edits["database_redactions"] = database_changes
                    check_file(actual, rel)
                    before = actual.stat()
                    sha = digest(actual)
                    archive.add(actual, arcname=rel, recursive=False)
                    after = actual.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("Source changed during packaging: " + rel)
                    members.append({"source": rel, "archive": "packages/" + final.name,
                                    "member": rel, "bytes": before.st_size, "sha256": sha,
                                    "consistent_sqlite_snapshot": actual == database,
                                    "backup_copy_changes": edits,
                                    "source_bytes": original.stat().st_size,
                                    "source_mtime_ns": original.stat().st_mtime_ns,
                                    "original_sha256": digest(original) if edits else sha})
            partial.replace(final)
            records.extend(members)
            write_json(receipt_path, {"package_sha256": digest(final), "members": members})
    adapter_records = []
    for adapter in inventory["final_adapters"]:
        original = source / adapter["path"]
        destination = models / "adapters" / adapter_destination(adapter["path"])
        destination.mkdir(parents=True, exist_ok=True)
        print("ADAPTER", str(destination.relative_to(models)), flush=True)
        for name in sorted(SIDECARS | {"adapter_model.safetensors"}):
            actual = original.parent / name
            if not actual.is_file():
                continue
            check_file(actual, str(actual.relative_to(source)))
            sha = digest(actual)
            if name == "adapter_model.safetensors" and sha != adapter["sha256"]:
                raise ValueError("Adapter changed since inventory: " + adapter["path"])
            output = destination / name
            shutil.copy2(actual, output)
            adapter_records.append({"source": str(actual.relative_to(source)),
                                    "path": str(output.relative_to(models)),
                                    "bytes": output.stat().st_size, "sha256": sha})
        if not (destination / "adapter_config.json").is_file():
            raise ValueError("Adapter configuration missing")
    assert len([r for r in adapter_records if r["path"].endswith("adapter_model.safetensors")]) == 23
    metadata = {"date": args.date, "source_git_commit": inventory["git_commit"],
                "source_inventory_sha256": digest(args.inventory),
                "scope": "Selected CoT research artifacts; not downloaded public data or base weights.",
                "restore_note": "Archive members retain original paths relative to the research root. Restore to a new directory. Adapters require their exact original base and tokenizer; full-model numerical restoration is not yet verified.",
                "source_datasets": [{k: v[k] for k in ("name", "source", "revision") if k in v}
                                    for v in inventory["open_sources"]]}
    write_json(data / "manifest.json", {**metadata, "members": records, "excluded": exclusions})
    write_json(models / "manifest.json", {**metadata, "files": adapter_records,
               "base_models": sorted({v["base_model_name_or_path"] for v in inventory["final_adapters"]})})
    for directory, title in [(data, "CoT generated datasets and experiment evidence"),
                             (models, "CoT final learned LoRA adapters")]:
        (directory / "README.md").write_text(
            "---\nlicense: other\n---\n# " + title + "\n\n"
            "Private backup prepared " + args.date + ". Author: 张瑞鹏 (ZrpChuang).\n\n"
            "Source code: https://github.com/ZrpChuang/CoT_Gen/tree/" + inventory["git_commit"] + "\n\n"
            "See manifest.json for source paths and checksums. No open source dataset or base model is re-uploaded. "
            "Outputs include research failures and historical experiments; this is not a claim of improvement or a new training run.\n")
    prepared = {"date": args.date, "source_git_commit": inventory["git_commit"], "repositories": {}}
    for name in ["dataset", "model"]:
        directory = target / name
        fs = [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size,
               "sha256": digest(p)} for p in sorted(directory.rglob("*")) if p.is_file()]
        prepared["repositories"][name] = {"bytes": sum(f["bytes"] for f in fs), "files": fs}
    write_json(target / "prepared.json", prepared)
    print("PREPARED", json.dumps({k: {"bytes": v["bytes"], "files": len(v["files"])}
                                  for k, v in prepared["repositories"].items()}), flush=True)


def publish(args):
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open
    api = HfApi()
    account = api.whoami()["name"]
    namespace = args.namespace or account
    prepared = json.loads((args.stage / "prepared.json").read_text())
    results = {"date": prepared["date"], "account": account, "repositories": {}}
    for kind, name in [("dataset", "CoT_Gen-artifacts-private"), ("model", "CoT_Gen-adapters-private")]:
        repo = namespace + "/" + name
        api.create_repo(repo_id=repo, repo_type=kind, private=True, exist_ok=True)
        if not api.repo_info(repo_id=repo, repo_type=kind).private:
            raise ValueError("Refusing to upload to a public repository: " + repo)
        folder = args.stage / kind
        for f in prepared["repositories"][kind]["files"]:
            if digest(folder / f["path"]) != f["sha256"]:
                raise ValueError("Local snapshot changed: " + f["path"])
        prefix = "snapshots/" + prepared["date"]
        existing = api.list_repo_files(repo_id=repo, repo_type=kind)
        expected_paths = {prefix + "/" + f["path"] for f in prepared["repositories"][kind]["files"]}
        existing_paths = {p for p in existing if p.startswith(prefix + "/")}
        if existing_paths:
            if existing_paths != expected_paths:
                raise ValueError("Existing dated snapshot has different files; will not overwrite: " + repo)
            revision = api.repo_info(repo_id=repo, repo_type=kind).sha
        else:
            commit = api.upload_folder(repo_id=repo, repo_type=kind, folder_path=folder,
                                       path_in_repo=prefix,
                                       commit_message="Back up reviewed CoT artifacts " + prepared["date"])
            revision = commit.oid
        files = {v.path: v for v in api.list_repo_tree(repo_id=repo, repo_type=kind,
                   revision=revision, recursive=True) if hasattr(v, "size")}
        checked = 0
        for record in prepared["repositories"][kind]["files"]:
            path = prefix + "/" + record["path"]
            remote = files.get(path)
            if remote is None or remote.size != record["bytes"]:
                raise ValueError("Remote size/missing-file mismatch: " + path)
            lfs = getattr(remote, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs else None
            if not sha:
                downloaded = hf_hub_download(repo_id=repo, repo_type=kind, revision=revision, filename=path)
                sha = digest(Path(downloaded))
            if sha != record["sha256"]:
                raise ValueError("Remote SHA256 mismatch: " + path)
            checked += 1
        result = {"repo_id": repo, "repo_type": kind, "private": True,
                  "revision": revision, "prefix": prefix, "verified_files": checked,
                  "bytes": prepared["repositories"][kind]["bytes"]}
        results["repositories"][kind] = result
        write_json(args.stage / "published.json", results)
        print("VERIFIED", json.dumps(result), flush=True)
    model = results["repositories"]["model"]
    filename = model["prefix"] + "/adapters/fresh_10k/NEU/Qwen_35_9B/adapter_model.safetensors"
    downloaded = hf_hub_download(repo_id=model["repo_id"], repo_type="model",
                                revision=model["revision"], filename=filename, force_download=True)
    with safe_open(downloaded, framework="np") as tensors:
        keys = list(tensors.keys())
        assert keys
        shapes = {key: list(tensors.get_slice(key).get_shape()) for key in keys}
    results["restore_check"] = {"downloaded_adapter": filename, "tensor_count": len(keys),
                               "header_shapes_read": True, "full_base_forward_test": False,
                               "sha256": digest(Path(downloaded)), "shapes": shapes}
    write_json(args.stage / "published.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--stage", type=Path, required=True)
    build.add_argument("--date", required=True)
    upload = commands.add_parser("publish")
    upload.add_argument("--stage", type=Path, required=True)
    upload.add_argument("--namespace")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else publish)(args)


if __name__ == "__main__":
    main()
