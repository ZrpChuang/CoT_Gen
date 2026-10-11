"""Hash validation and explicit restoration of archived experiment evidence."""

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

from .settings import ROOT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate():
    records = json.loads((ROOT / "configs/integrity.json").read_text())
    for row in records:
        path = ROOT / row["target"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise ValueError(f"Preserved evidence changed: {row['target']}")
    return {"verified_files": len(records), "passed": True}


def restore(archive, destination):
    """Restore regular files into a new directory; record links for review."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Restore destination must be a new directory")
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Restore outside the maintained research directory")
    destination.mkdir(parents=True)
    links = []
    observed = {}
    manifest_seen = False
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.name == "MANIFEST.jsonl":
                expected = [
                    json.loads(line)
                    for line in source.extractfile(member).read().decode().split("\n")
                    if line
                ]
                if len(expected) != len(observed):
                    raise ValueError("Archive member count differs from its manifest")
                for row in expected:
                    for key, value in observed[row["path"]].items():
                        if row[key] != value:
                            raise ValueError(f"Archive hash mismatch: {row['path']}")
                manifest_seen = True
                continue
            if member.issym():
                links.append({"path": member.name, "target": member.linkname})
                observed[member.name] = {"type": "symlink", "target": member.linkname}
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with target.open("xb") as out:
                handle = source.extractfile(member)
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    out.write(chunk)
                    digest.update(chunk)
            observed[member.name] = {
                "type": "file",
                "sha256": digest.hexdigest(),
                "bytes": member.size,
            }
    if not manifest_seen:
        raise ValueError("Archive manifest is missing")
    (destination / "RESTORED_LINKS.json").write_text(json.dumps(links, indent=2) + "\n")
    return {"destination": str(destination), "recorded_links": len(links)}
