"""Windows-only repair step: give torch and onnxruntime a modern Visual C++ runtime.

Why this is needed
------------------
torch 2.x and onnxruntime link against the Visual C++ 2015-2022 runtime. If the
machine only has an older redistributable installed (14.13 from VS 2017 is
common, and it is missing `msvcp140_2.dll` entirely), importing them fails with:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading ...\\torch\\lib\\c10.dll

    ImportError: DLL load failed while importing onnxruntime_pybind11_state:
    A dynamic link library (DLL) initialization routine failed.

Both matter here. torch powers the local embeddings; onnxruntime is never used
for anything, but chromadb instantiates its default ONNX embedding function at
class-definition time, so the package has to be importable or `import chromadb`
raises.

The `msvc-runtime` package ships the current redistributable DLLs into the
virtual environment. This script copies them next to each native extension,
which is the first directory Windows searches when resolving that DLL's
dependencies. It changes nothing outside the project — no system install, no
registry, no admin.

Run once after `pip install -r requirements.txt`, or any time you reinstall
torch or onnxruntime:

    python setup_windows_runtime.py

If both already import, this script says so and does nothing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

RUNTIME_DLLS = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
)


def _native_dir(relative: str, marker: str) -> Path | None:
    """Find the directory holding a package's native extension."""
    for base in Path(sys.prefix).rglob(relative):
        if (base / marker).exists():
            return base
    return None


def _source_dirs() -> list[Path]:
    prefix = Path(sys.prefix)
    return [prefix, prefix / "Scripts", Path(sys.base_prefix)]


def _check(module: str) -> bool:
    """Return True if the module imports cleanly."""
    try:
        __import__(module)
        return True
    except (OSError, ImportError):
        return False


def _patch(target: Path) -> int:
    copied = 0
    for name in RUNTIME_DLLS:
        for source_dir in _source_dirs():
            source = source_dir / name
            if source.exists():
                shutil.copy2(source, target / name)
                copied += 1
                break
    return copied


TARGETS = (
    ("torch", "torch/lib", "c10.dll"),
    ("onnxruntime", "onnxruntime/capi", "onnxruntime_pybind11_state.pyd"),
)


def main() -> int:
    if sys.platform != "win32":
        print("Not Windows — nothing to do.")
        return 0

    failures = 0
    for module, relative, marker in TARGETS:
        if _check(module):
            print(f"{module}: imports fine — nothing to do.")
            continue

        target = _native_dir(relative, marker)
        if target is None:
            print(f"{module}: not installed — run: pip install -r requirements.txt")
            failures += 1
            continue

        copied = _patch(target)
        if copied == 0:
            print(
                f"{module}: no runtime DLLs found in the environment.\n"
                "  Install them into the venv with:  pip install msvc-runtime\n"
                "  or install the Microsoft Visual C++ 2015-2022 Redistributable (x64)."
            )
            failures += 1
            continue

        print(f"{module}: copied {copied} runtime DLLs into {target}")

    if failures:
        return 1

    print("\nDone. Re-run this script to confirm both modules now import.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
