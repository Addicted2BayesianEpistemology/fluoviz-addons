#!/usr/bin/env python3
"""Build the downloadable files of the add-ons of this repository (standard library only).

    python scripts/build_addon.py                    # every add-on in addons/  ->  dist/
    python scripts/build_addon.py autoencoders       # one add-on
    python scripts/build_addon.py --out some/folder

For every add-on it writes ``dist/<id>-<version>.zip`` (ONE top-level folder ``<id>-<version>/`` with the manifest, the Python package, ``web/`` and ``docs/``; no tests,
no caches) and ONE ``dist/index.json`` that lists all of them (sha256, size, the manifest): the index is what the FluoViz **Install** button reads, the zip is what it downloads
(``url`` is relative to the index, so the folder can be copied to any file server or release page, or used from a USB stick: Settings > Add-ons > Where add-ons come from).
The format is specified in FluoViz ``docs/modules/addons.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", "tests", ".pytest_cache", ".git", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".orig", ".rej", ".swp", ".DS_Store"}
SKIP_FILES = {"PROMPT.md"}                       # the long implementation briefs stay in the repository, they are not part of the add-on
FIXED_TIME = (2026, 1, 1, 0, 0, 0)                  # the same files give the same zip (and the same sha256)


def files_of(folder: Path):
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if p.is_dir() or any(part in SKIP_DIRS for part in rel.parts) or p.suffix in SKIP_SUFFIXES or p.name in SKIP_SUFFIXES or (len(rel.parts) == 1 and p.name in SKIP_FILES):
            continue
        yield p, rel


def build_one(folder: Path, out: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    ident, version = manifest["id"], manifest["version"]
    for key in ("id", "version", "name", "summary", "python_package", "entry", "requires_fluoviz"):
        if not manifest.get(key):
            raise SystemExit(f"{folder}: manifest.json has no '{key}'")
    if not (folder / manifest["python_package"] / "__init__.py").is_file():
        raise SystemExit(f"{folder}: the Python package '{manifest['python_package']}' is missing")
    top = f"{ident}-{version}"
    zpath = out / f"{top}.zip"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, rel in files_of(folder):
            info = zipfile.ZipInfo(f"{top}/{rel.as_posix()}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    data = zpath.read_bytes()
    return {
        "id": ident, "name": manifest["name"], "summary": manifest["summary"], "version": version,
        "url": zpath.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
        "requires_fluoviz": manifest["requires_fluoviz"], "platforms": manifest.get("platforms", []),
        "extra_download_mb": manifest.get("extra_download_mb", 0), "manifest": manifest,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ids", nargs="*", help="add-on ids (default: all)")
    ap.add_argument("--out", default=str(ROOT / "dist"), help="output folder (default: dist/)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    folders = sorted(p for p in (ROOT / "addons").iterdir() if (p / "manifest.json").is_file())
    if args.ids:
        folders = [p for p in folders if p.name in args.ids]
        missing = set(args.ids) - {p.name for p in folders}
        if missing:
            raise SystemExit("no such add-on: " + ", ".join(sorted(missing)))
    entries = [build_one(f, out) for f in folders]
    index = {"schema": 1, "addons": entries}
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for e in entries:
        print(f"{e['id']} {e['version']}: {out / e['url']}  ({e['size_bytes'] / 1e6:.2f} MB, sha256 {e['sha256'][:12]}...)")
    print(f"index: {out / 'index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
