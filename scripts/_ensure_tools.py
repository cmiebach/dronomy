"""Provision the external tools the pipeline shells out to, on a fresh machine.

Only exiftool needs provisioning (it decodes the DJI `djmd` GPS stream for the
ground-truth track). On Windows we fetch the official portable build from
exiftool.org into `tools/exiftool/` — no admin, no PATH surgery — and point the
EXIFTOOL env var at it (which `telemetry.find_exiftool` already honours). If it
is already installed (PATH / env / a prior provision), this is a no-op.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)
from dronomy_loc.data.telemetry import find_exiftool
from dronomy_loc.data.fetch import download_file


def ensure_exiftool(tools_dir: str | Path = "tools", *, verbose: bool = True) -> str:
    """Return a path to a runnable exiftool, provisioning a portable copy if needed."""
    found = find_exiftool()
    if found:
        if verbose:
            print(f"exiftool: found at {found}")
        return found

    tools_dir = Path(tools_dir)
    local = tools_dir / "exiftool" / "exiftool.exe"
    if local.is_file():
        os.environ["EXIFTOOL"] = str(local)
        if verbose:
            print(f"exiftool: using portable copy {local}")
        return str(local)

    if os.name != "nt":
        raise RuntimeError(
            "exiftool not found. Install it (e.g. `sudo apt-get install -y "
            "libimage-exiftool-perl` or `brew install exiftool`) and re-run.")

    import requests
    ver = requests.get("https://exiftool.org/ver.txt", timeout=30).text.strip()
    # The Windows build is hosted on SourceForge; the `/download` URL redirects
    # to a mirror serving the zip (exiftool.org itself only has the source tar).
    url = (f"https://sourceforge.net/projects/exiftool/files/"
           f"exiftool-{ver}_64.zip/download")
    tools_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tools_dir / f"exiftool-{ver}_64.zip"
    if verbose:
        print(f"exiftool: not found — downloading portable v{ver} ...")
    download_file(url, zip_path)

    extract = tools_dir / "_exiftool_extract"
    if extract.exists():
        shutil.rmtree(extract, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract)

    # The zip ships `exiftool(-k).exe` (interactive-pause build) beside an
    # `exiftool_files/` dir. Rename the exe to `exiftool.exe` to disable the
    # "-- press Enter --" pause, and keep exiftool_files alongside it.
    exe_src = next(extract.rglob("exiftool(-k).exe"), None) \
        or next(extract.rglob("exiftool.exe"), None)
    if exe_src is None:
        raise RuntimeError(f"no exiftool exe inside {zip_path}")
    files_src = next((p for p in extract.rglob("exiftool_files") if p.is_dir()), None)

    dest = tools_dir / "exiftool"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(exe_src, dest / "exiftool.exe")
    if files_src:
        shutil.copytree(files_src, dest / "exiftool_files", dirs_exist_ok=True)
    shutil.rmtree(extract, ignore_errors=True)

    os.environ["EXIFTOOL"] = str(dest / "exiftool.exe")
    if verbose:
        print(f"exiftool: provisioned -> {dest / 'exiftool.exe'}")
    return str(dest / "exiftool.exe")


if __name__ == "__main__":
    print(ensure_exiftool())
