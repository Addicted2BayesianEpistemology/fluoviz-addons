#!/usr/bin/env python3
"""Build the downloadable files of the add-ons of this repository (standard library only).

    python scripts/build_addon.py                    # every add-on in addons/  ->  dist/
    python scripts/build_addon.py autoencoders       # one add-on
    python scripts/build_addon.py --out some/folder
    python scripts/build_addon.py --keep-from old/                 # also list the add-ons of an earlier release (a folder with its index.json and zips)
    python scripts/build_addon.py --keep-release v1.1.0            # the same, the files fetched with `gh release download v1.1.0`
    python scripts/build_addon.py --keep-release latest            # the same, from the release that is "latest" now

For every add-on it writes ``dist/<id>-<version>.zip`` (ONE top-level folder ``<id>-<version>/`` with the manifest, the Python package, ``web/`` and ``docs/``; no tests,
no caches) and ONE ``dist/index.json`` that lists all of them (sha256, size, the manifest): the index is what the FluoViz **Install** button reads, the zip is what it downloads
(``url`` is relative to the index, so the folder can be copied to any file server or release page, or used from a USB stick: Settings > Add-ons > Where add-ons come from).
The format is specified in FluoViz ``docs/modules/addons.md``.

**Several versions of one add-on.**  The index may list an add-on more than once; FluoViz offers, per add-on, the NEWEST version that fits this computer and this FluoViz
(``requires_fluoviz``), and resolves ``requires_addons`` the same way (``fluoviz/addons/sources.py: Index.find``, ``deps.py: find_fitting``).  So when an add-on starts to need a
newer FluoViz, the release keeps the entries of the previous release (``--keep-from`` / ``--keep-release``): a person with the older FluoViz is still offered what runs there.
A kept entry is copied with its zip (checked against its sha256); it is left out when this build makes the same id and version (the new build wins) or when a newer version of
the same add-on asks for exactly the same FluoViz versions (it would never be offered).  Kept entries are kept again by the next release that keeps this one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
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


def _vkey(v: str) -> tuple:
    """A sort key of a version '1.2.0' / '1.2.0-rc1' (a pre-release sorts before its release)."""
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$", str(v))
    if not m:
        return (-1, -1, -1, 0, str(v))
    return (int(m[1]), int(m[2]), int(m[3]), 0 if m[4] else 1, m[4] or "")


def fetch_release(tag: str, into: Path) -> Path:
    """Download index.json and the zips of a release of this repository with the GitHub command-line tool (``gh``); ``latest`` = the release marked latest."""
    if tag == "latest":
        tag = subprocess.run(["gh", "api", "repos/{owner}/{repo}/releases/latest", "--jq", ".tag_name"], check=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    into.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", tag, "--pattern", "index.json", "--pattern", "*.zip", "--dir", str(into), "--clobber"], check=True, cwd=ROOT)
    print(f"kept from release {tag}: {into}")
    return into


def kept_entries(folder: Path, out: Path) -> list[dict]:
    """The entries of the index in ``folder``, their zips copied to ``out`` (each checked against its sha256)."""
    idx = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    entries = []
    for e in idx.get("addons", []):
        src = folder / e["url"]
        if "/" in e["url"] or "\\" in e["url"] or not src.is_file():
            raise SystemExit(f"{folder}: the zip of {e['id']} {e['version']} ('{e['url']}') is not in that folder")
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            raise SystemExit(f"{folder}: the zip of {e['id']} {e['version']} does not match its sha256")
        out.mkdir(parents=True, exist_ok=True)
        if src.resolve() != (out / src.name).resolve():
            shutil.copyfile(src, out / src.name)
        entries.append(e)
    return entries


def merge(built: list[dict], kept: list[dict]) -> list[dict]:
    """The new entries plus the kept ones that can still be offered to someone; per add-on newest first."""
    have = {(e["id"], e["version"]) for e in built}
    out = list(built)
    for e in kept:
        if (e["id"], e["version"]) in have:
            continue                                                    # the new build of the same version wins
        newer_same_range = any(x["id"] == e["id"] and _vkey(x["version"]) > _vkey(e["version"]) and x.get("requires_fluoviz") == e.get("requires_fluoviz") for x in out + kept)
        if newer_same_range:
            continue                                                    # never offered: a newer version serves exactly the same FluoViz
        have.add((e["id"], e["version"]))
        out.append(e)
    out.sort(key=lambda e: (e["id"], tuple(-x if isinstance(x, int) else 0 for x in _vkey(e["version"])[:4])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ids", nargs="*", help="add-on ids (default: all)")
    ap.add_argument("--out", default=str(ROOT / "dist"), help="output folder (default: dist/)")
    ap.add_argument("--keep-from", action="append", default=[], metavar="FOLDER", help="also list the add-ons of an earlier release: a folder with its index.json and zips")
    ap.add_argument("--keep-release", action="append", default=[], metavar="TAG", help="the same, downloaded with gh from the release TAG of this repository ('latest' = the latest release)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    folders = sorted(p for p in (ROOT / "addons").iterdir() if (p / "manifest.json").is_file())
    if args.ids:
        folders = [p for p in folders if p.name in args.ids]
        missing = set(args.ids) - {p.name for p in folders}
        if missing:
            raise SystemExit("no such add-on: " + ", ".join(sorted(missing)))
    entries = [build_one(f, out) for f in folders]
    kept = []
    for folder in args.keep_from:
        kept += kept_entries(Path(folder), out)
    for tag in args.keep_release:
        with tempfile.TemporaryDirectory() as tmp:
            kept += kept_entries(fetch_release(tag, Path(tmp)), out)
    entries = merge(entries, kept)
    for e in entries:                                                   # every zip the index names is next to it
        if not (out / e["url"]).is_file():
            raise SystemExit(f"{e['id']} {e['version']}: {e['url']} is missing in {out}")
    index = {"schema": 1, "addons": entries}
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for e in entries:
        print(f"{e['id']} {e['version']}: {out / e['url']}  ({e['size_bytes'] / 1e6:.2f} MB, sha256 {e['sha256'][:12]}...)")
    print(f"index: {out / 'index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
