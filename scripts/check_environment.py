#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""检查安装版本与依赖偏离；0.31 基线没有已记录的 override，任何冲突都算失败。"""
import importlib.metadata as metadata
import sys
from packaging.requirements import Requirement


def check():
    failures = []
    for dist in metadata.distributions():
        name = canonical_name(dist.metadata["Name"])
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            try:
                version = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                failures.append(f"{name}: missing {req.name}")
                continue
            if req.specifier.contains(version, prereleases=True):
                continue
            failures.append(f"{name}: requires {req.name}{req.specifier}, installed {version}")
    for name, expected in {"vllm": "0.31.0", "torch": "2.13.0",
                           "transformers": "5.15.1", "flashinfer-python": "0.7.0.post1",
                           "flashinfer-cubin": "0.7.0.post1"}.items():
        got = metadata.version(name).split("+")[0]
        if got != expected:
            failures.append(f"{name}: expected {expected}, found {got}")
    for name in ("vllm", "torch", "transformers", "tokenizers",
                 "flashinfer-python", "flashinfer-cubin", "nccl-extensions"):
        print(name, metadata.version(name))
    for item in failures:
        print("ERROR:", item)
    return 1 if failures else 0


def canonical_name(value):
    return value.lower().replace("_", "-")


if __name__ == "__main__":
    sys.exit(check())
