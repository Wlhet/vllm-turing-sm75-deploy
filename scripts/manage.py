#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Linux 专用进程管理：仅管理本项目启动并记录身份的进程，不扫描/杀死其他服务。"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from common import ROOT, build_command, health_url, load_config, runtime_env, validate

RUNTIME = ROOT / ".runtime"
STATE = RUNTIME / "server.json"
_CHILDREN = {}


def process_identity(pid):
    """PID + /proc 启动时钟 + boot_id 避免 PID 重用时误停其他进程。"""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def owned(state):
    identity = process_identity(state.get("pid", -1))
    if not identity or any(identity[k] != state.get(k) for k in identity):
        return False
    try:
        env = Path(f"/proc/{identity['pid']}/environ").read_bytes().split(b"\0")
        return f"SM75_PROJECT_ROOT={ROOT}".encode() in env and os.getpgid(identity["pid"]) == identity["pid"]
    except OSError:
        return False


def read_state():
    try:
        state = json.loads(STATE.read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


@contextmanager
def locked():
    import fcntl
    RUNTIME.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (RUNTIME / "manage.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def port_available(cfg):
    family = socket.AF_INET6 if ":" in cfg["HOST"] else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((cfg["HOST"], int(cfg["PORT"])))


def start(cfg):
    validate(cfg, require_files=True)
    with locked():
        state = read_state()
        if owned(state):
            raise RuntimeError(f"Already running: PID {state['pid']}")
        port_available(cfg)
        logs = RUNTIME / "logs"
        logs.mkdir(mode=0o700, exist_ok=True)
        logfile = logs / f"server-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}.log"
        with logfile.open("x") as log:
            child = subprocess.Popen(build_command(cfg), env=runtime_env(cfg), cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        _CHILDREN[child.pid] = child
        state = process_identity(child.pid)
        if state is None:
            raise RuntimeError(f"Process exited; inspect {logfile.relative_to(ROOT)}")
        state["log"] = str(logfile.relative_to(ROOT))
        temp = STATE.with_suffix(".tmp")
        temp.write_text(json.dumps(state) + "\n")
        os.replace(temp, STATE)
        time.sleep(0.3)
        if child.poll() is not None:
            raise RuntimeError(f"Startup failed; inspect {logfile.relative_to(ROOT)}")
        print(f"Launched PID {child.pid}; NOT necessarily ready. Log: {state['log']}")
        print("Next: python3 scripts/manage.py wait --timeout 1800")


def stop(timeout):
    with locked():
        state = read_state()
        if not owned(state):
            print("No owned running process; no signals sent.")
            return
        os.killpg(state["pid"], signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while owned(state) and time.monotonic() < deadline:
            time.sleep(0.2)
        if owned(state):
            raise RuntimeError("Graceful stop timed out; no SIGKILL sent. Inspect manually.")
        child = _CHILDREN.pop(state["pid"], None)
        if child is not None:
            child.wait(timeout=2)
        STATE.unlink(missing_ok=True)
        print("Stopped.")


def healthy(cfg):
    try:
        # 直连本机，不把健康检查交给 shell 中继承的 HTTP 代理。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(health_url(cfg) + "/health", timeout=3) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "stop", "status", "wait", "health", "logs", "print-command"))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    os.umask(0o077)
    cfg = load_config()
    if args.action == "print-command":
        validate(cfg)
        print(shlex.join(build_command(cfg)))
        print("# VLLM_API_KEY is supplied through the environment, not printed.")
    elif args.action == "start":
        start(cfg)
    elif args.action == "run":
        validate(cfg, require_files=True)
        RUNTIME.mkdir(mode=0o700, parents=True, exist_ok=True)
        port_available(cfg)
        os.execve(build_command(cfg)[0], build_command(cfg), runtime_env(cfg))
    elif args.action == "stop":
        stop(args.timeout)
    elif args.action == "status":
        state = read_state()
        print(json.dumps({"owned_process_running": owned(state), "http_healthy": healthy(cfg),
                          "pid": state.get("pid"), "log": state.get("log")}, indent=2))
        return 0 if owned(state) else 1
    elif args.action == "health":
        ok = healthy(cfg)
        print("healthy" if ok else "unavailable")
        return 0 if ok else 1
    elif args.action == "logs":
        value = read_state().get("log")
        if not value:
            raise RuntimeError("No recorded log")
        log = (ROOT / value).resolve()
        if log.parent != (RUNTIME / "logs").resolve():
            raise RuntimeError("Invalid log path")
        subprocess.run(["tail", "-n", "80", "-f", str(log)], check=True)
    elif args.action == "wait":
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if not owned(read_state()):
                raise RuntimeError("Owned process exited or was not started by manage.py start")
            if healthy(cfg):
                print("Ready.")
                return 0
            time.sleep(2)
        raise RuntimeError("Readiness timeout; process left running. Inspect logs.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        # 不输出环境或完整启动参数，避免认证信息流入管理日志。
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
