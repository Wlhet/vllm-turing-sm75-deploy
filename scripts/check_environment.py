#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""检查安装版本与依赖偏离；只允许已记录的 FlashInfer 覆盖，不掩盖其他冲突。"""
import importlib.metadata as metadata
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check():
    failures = []
    allowed = []
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
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
            message = f"{name}: requires {req.name}{req.specifier}, installed {version}"
            if (name == "vllm" and dist.version == "0.30.0"
                    and canonicalize_name(req.name) == "flashinfer-python"
                    and str(req.specifier) == "==0.6.18.post1" and version == "0.7.0"):
                allowed.append(message)
            else:
                failures.append(message)
    for item in allowed:
        print("DOCUMENTED OVERRIDE:", item)
    for item in failures:
        print("ERROR:", item)
    for name in ("vllm", "torch", "transformers", "flashinfer-python", "flashinfer-cubin"):
        print(name, metadata.version(name))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(check())
