#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""小规模 OpenAI 兼容 API 验证；会实际生成 token，不是性能基准。"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from common import health_url, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="OpenAI URL including /v1; defaults to configured backend")
    parser.add_argument("--key-env", default="VLLM_API_KEY", help="For New API use NEWAPI_API_KEY")
    parser.add_argument("--model", help="Defaults to SERVED_MODEL_NAME")
    parser.add_argument("--allow-remote", action="store_true", help="Explicitly allow sending a key to a remote host")
    args = parser.parse_args()
    cfg = load_config()
    base = (args.base_url or health_url(cfg) + "/v1").rstrip("/")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Invalid API URL")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        raise ValueError("Remote endpoint requires --allow-remote; use TLS outside a trusted network")
    key = os.environ.get(args.key_env) or (cfg["VLLM_API_KEY"] if args.key_env == "VLLM_API_KEY" else "")
    if not key or key.startswith("replace-"):
        raise ValueError("Missing API key; provide it through the selected environment variable")
    # 不跟随重定向，避免认证头被转发到意外主机。
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({}))
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    with opener.open(urllib.request.Request(base + "/models", headers=headers), timeout=20) as response:
        models = json.load(response)
    model = args.model or cfg["SERVED_MODEL_NAME"]
    if model not in [x["id"] for x in models["data"]]:
        raise RuntimeError("Requested model is absent from /models")
    payload = {"model": model, "messages": [{"role": "user", "content": "Reply with READY only."}],
               "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
    with opener.open(req, timeout=180) as response:
        result = json.load(response)
    content = result["choices"][0]["message"].get("content", "")
    if not content.strip():
        raise RuntimeError("Empty model response")
    print(json.dumps({"ok": True, "model": model, "reply": content, "usage": result.get("usage")}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        # 服务端错误体可能包含敏感数据，因此仅打印状态码。
        print(f"HTTP error {exc.code}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
