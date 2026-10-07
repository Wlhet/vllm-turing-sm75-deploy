#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""在本项目私有虚拟环境应用固定版本 overlay，预编译 SM75 扩展。"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import ROOT, load_config, resolved, runtime_env


def preflight(cfg):
    if sys.platform != "linux":
        raise RuntimeError("Linux x86_64 with CUDA is required")
    import platform
    if platform.machine() != "x86_64":
        raise RuntimeError("Only Linux x86_64 has been validated")
    tools = (str(resolved(cfg["CUDA_HOME"]) / "bin/nvcc"), cfg["CC"], cfg["CXX"], "nvidia-smi")
    for tool in tools:
        if not shutil.which(tool):
            raise RuntimeError(f"Missing prerequisite: {tool}")
    print("Native prerequisites found (driver/toolchain versions still need operator review).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    preflight(cfg)
    if args.preflight:
        return
    venv = ROOT / ".venv"
    if venv.is_symlink() or Path(sys.prefix).resolve() != venv.resolve():
        raise RuntimeError("Use this project's .venv/bin/python; refusing to modify another environment")
    for name, expected in {"vllm": "0.30.0", "transformers": "5.15.1", "torch": "2.13.0"}.items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise RuntimeError(f"Unexpected {name} version")
    for name in ("PYTHONHOME", "PYTHONPATH"):
        os.environ.pop(name, None)
    os.environ.update(runtime_env(cfg))
    vendor = ROOT / "vendor/vllm-turing"
    evidence = ROOT / ".runtime/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(vendor / "docker/patch_transformers_startup.py")], check=True)
    subprocess.run([sys.executable, str(vendor / "docker/install_sm75_overlay.py"),
                    str(vendor / "vllm"), "--source-copy", str(evidence / "source/vllm")], check=True)
    import vllm
    package = Path(vllm.__file__).resolve().parent
    if not package.is_relative_to(venv.resolve()):
        raise RuntimeError("vLLM import escaped project virtualenv")
    sys.path.insert(0, str(vendor / "docker"))
    from install_speculative import install
    install(vendor / "docker/speculative", package, evidence / "speculative-files.json")
    subprocess.run([sys.executable, str(vendor / "vllm/third_party/flash_qla_sm75/build_extension.py"),
                    "--build-directory", str(ROOT / ".runtime/cache/qla"),
                    "--output", str(ROOT / ".runtime/extensions/flash_qla_sm75_gdn.so"), "--verbose"], check=True)
    (evidence / "installation.json").write_text(json.dumps({
        "upstream_commit": json.loads((ROOT / "UPSTREAM.json").read_text())["commit"],
        "versions": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers", "flashinfer-python")},
    }, indent=2) + "\n")
    print("Overlay installed; model has NOT been started.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Installation failed: {exc}")
