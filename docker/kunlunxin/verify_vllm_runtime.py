# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot/verify every installed vLLM file, including non-source resources."""

import argparse
import hashlib
import json
import stat
from pathlib import Path


def snapshot(root: Path) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Not a package directory: {root}")
    entries = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        entry = {"mode": stat.S_IMODE(mode)}
        if stat.S_ISLNK(mode):
            entry.update(type="symlink", target=str(path.readlink()))
        elif stat.S_ISDIR(mode):
            entry.update(type="directory")
        elif stat.S_ISREG(mode):
            entry.update(
                type="file", sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        else:
            raise ValueError(f"Unsupported package entry: {path}")
        entries[str(path.relative_to(root))] = entry
    return {"root_mode": stat.S_IMODE(root.stat().st_mode), "entries": entries}


def verify(root: Path, expected: dict) -> None:
    actual = snapshot(root)
    if actual != expected:
        before, after = expected["entries"], actual["entries"]
        differences = {
            "missing": sorted(before.keys() - after.keys()),
            "extra": sorted(after.keys() - before.keys()),
            "changed": sorted(
                k for k in before.keys() & after.keys() if before[k] != after[k]
            ),
            "root_mode_changed": expected["root_mode"] != actual["root_mode"],
        }
        raise ValueError(json.dumps(differences, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, help="Verify instead of snapshotting")
    args = parser.parse_args()
    if args.manifest:
        verify(args.root, json.loads(args.manifest.read_text()))
        print("Installed vLLM tree matches the complete original manifest")
    else:
        print(json.dumps(snapshot(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
