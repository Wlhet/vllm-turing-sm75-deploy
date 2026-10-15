#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""发布前检查源码语法、上游完整性、禁入文件和常见敏感信息；不导入 GPU 依赖。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", ".runtime", "__pycache__", "dist"}
FORBIDDEN = {".safetensors", ".gguf", ".whl", ".so", ".bin", ".pt", ".pth", ".log", ".pem", ".key", ".zip", ".gz"}


def problems(path, data):
    issues = []
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        issues.append("private environment file")
    if path.suffix in FORBIDDEN:
        issues.append("runtime/binary/secret artifact")
    if len(data) > 5 * 1024 * 1024:
        issues.append("file exceeds source-only size limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return issues + ["non-UTF-8/binary file"]
    patterns = {
        "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "token-shaped string": r"\b(?:sk-|hf_)[A-Za-z0-9_-]{20,}",
        "private endpoint": r"https?://(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?=[:/\s'\"]|$)",
        "personal home path": r"/home/[a-zA-Z0-9_-]+/",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text):
            issues.append(label)
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError:
            issues.append("invalid Python syntax")
    if b"\r\n" in data:
        issues.append("CRLF; expected LF")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="Inspect Git index candidates; private .env may remain untracked")
    args = parser.parse_args()
    if args.tracked:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        paths = [ROOT / p.decode() for p in result.stdout.split(b"\0") if p]
        if not paths:
            raise RuntimeError("Empty Git index; stage the intended public files first")
        # 工作树必须与暂存内容相同，避免审查的是A却发布了B。
        subprocess.run(["git", "diff", "--exit-code", "--quiet"], cwd=ROOT, check=True)
    else:
        paths = [p for p in ROOT.rglob("*") if p.is_file() and not any(x in IGNORED for x in p.relative_to(ROOT).parts)]
    failures = []
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            failures.append(f"{relative}: symlink forbidden")
            continue
        for issue in problems(relative, path.read_bytes()):
            failures.append(f"{relative}: {issue}")
    metadata = json.loads((ROOT / "UPSTREAM.json").read_text())
    vendor = ROOT / "vendor/vllm-turing"
    for relative, original in metadata["original_sha256"].items():
        expected = metadata.get("local_changes", {}).get(relative, {}).get("sha256", original)
        path = vendor / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"vendor/{relative}: provenance checksum mismatch")
    if args.tracked:
        tracked = {p.relative_to(ROOT).as_posix() for p in paths}
        required = {"vendor/vllm-turing/" + name for name in metadata["original_sha256"]}
        for missing in sorted(required - tracked):
            failures.append(f"{missing}: required vendor file is not staged for publication")
    expected_paths = set(metadata["original_sha256"])
    actual_paths = {p.relative_to(vendor).as_posix() for p in vendor.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if expected_paths != actual_paths:
        failures.append("Vendored file inventory differs from UPSTREAM.json")
    for relative, entry in metadata.get("project_local", {}).items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"{relative}: project local checksum mismatch")
    for item in failures:
        print("FAIL:", item)
    print(f"Checked {len(paths)} public source files; {len(failures)} problem(s).")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
