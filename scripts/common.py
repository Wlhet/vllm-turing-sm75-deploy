#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""原生部署的共享配置与参数构建；不导入 torch，不触碰既有部署。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "MODEL_PATH": "", "VLLM_API_KEY": "", "SERVED_MODEL_NAMES": "Qwen3.8-27B",
    "HOST": "127.0.0.1", "PORT": "8000", "GPU_IDS": "0,1,2,3",
    "TENSOR_PARALLEL_SIZE": "4", "MAX_MODEL_LEN": "262144",
    "GPU_MEMORY_UTILIZATION": "0.92", "MAX_NUM_SEQS": "4",
    "MAX_NUM_BATCHED_TOKENS": "4096", "LONG_PREFILL_TOKEN_THRESHOLD": "2048",
    "MTP_TOKENS": "3", "QUANTIZATION": "modelopt_mixed",
    "FIREFLY_DIRECT": "1", "FIREFLY_AR": "1",
    "FIREFLY_AR_BACKEND": "auto", "CUDA_HOME": "/usr/local/cuda-13.1",
    "CC": "gcc-13", "CXX": "g++-13", "MAX_JOBS": "2",
}


def load_config(path=None, environ=None):
    """严格读取白名单 KEY=value；绝不 source .env，避免执行其中的代码。"""
    cfg = dict(DEFAULTS)
    path = Path(path) if path else ROOT / ".env"
    if path.exists():
        for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            key = key.strip()
            if not sep or key not in DEFAULTS:
                raise ValueError(f"Invalid config key at line {number}; see .env.example")
            values = shlex.split(value, comments=True, posix=True)
            if len(values) > 1:
                raise ValueError(f"Quote whitespace at config line {number}")
            cfg[key] = values[0] if values else ""
    env = os.environ if environ is None else environ
    cfg.update({key: env[key] for key in DEFAULTS if key in env})
    return cfg


def resolved(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def validate(cfg, require_files=False):
    """在启动前拒绝危险默认值、非法数值与不完整模型路径。"""
    key = cfg["VLLM_API_KEY"]
    if len(key) < 16 or key.startswith("replace-") or "\n" in key or "\r" in key:
        raise ValueError("Set VLLM_API_KEY to a real secret of at least 16 characters")
    if not cfg["MODEL_PATH"] or cfg["MODEL_PATH"].startswith("/path/to/"):
        raise ValueError("Set MODEL_PATH to the local model directory")
    if not cfg["HOST"] or cfg["HOST"].startswith("-"):
        raise ValueError("Invalid HOST")
    names = shlex.split(cfg["SERVED_MODEL_NAMES"], comments=True)
    if not names or any(not n or n.startswith("-") for n in names):
        raise ValueError("SERVED_MODEL_NAMES must be one or more plain names")
    if cfg["QUANTIZATION"] and (" " in cfg["QUANTIZATION"] or cfg["QUANTIZATION"].startswith("-")):
        raise ValueError("QUANTIZATION must be a single token or empty to let vLLM auto-detect")
    positive = ("PORT", "TENSOR_PARALLEL_SIZE", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
                "MAX_NUM_BATCHED_TOKENS", "LONG_PREFILL_TOKEN_THRESHOLD", "MAX_JOBS")
    for name in positive:
        if int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if int(cfg["PORT"]) > 65535:
        raise ValueError("Invalid PORT")
    if not 0 < float(cfg["GPU_MEMORY_UTILIZATION"]) < 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be between 0 and 1")
    if not 0 <= int(cfg["MTP_TOKENS"]) <= 8:
        raise ValueError("MTP_TOKENS must be 0..8; 0 disables MTP")
    ids = cfg["GPU_IDS"].split(",")
    if any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("GPU_IDS must be distinct comma-separated GPU indices")
    if len(ids) != int(cfg["TENSOR_PARALLEL_SIZE"]):
        raise ValueError("GPU_IDS count must match TENSOR_PARALLEL_SIZE")
    if int(cfg["LONG_PREFILL_TOKEN_THRESHOLD"]) > int(cfg["MAX_NUM_BATCHED_TOKENS"]):
        raise ValueError("Prefill threshold exceeds batch token budget")
    if cfg["FIREFLY_AR_BACKEND"] not in ("auto", "p2p", "shm"):
        raise ValueError("Invalid FIREFLY_AR_BACKEND")
    for name in ("FIREFLY_DIRECT", "FIREFLY_AR"):
        if cfg[name] not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
    if require_files:
        if not (resolved(cfg["MODEL_PATH"]) / "config.json").is_file():
            raise ValueError("MODEL_PATH must contain config.json")
        if not (ROOT / ".venv/bin/vllm").is_file():
            raise ValueError("Run bash scripts/install.sh first")
        if not (ROOT / ".runtime/extensions/flash_qla_sm75_gdn.so").is_file():
            raise ValueError("Missing FlashQLA extension; finish installation first")


def build_command(cfg):
    """只生成 argv，不用 shell 拼接；密钥通过环境变量传递，不放入命令行。

    HYBRID 结构模型（NVFP4 W4A16 等）需要显式量化器，见 .env.example。"""
    command = [str(ROOT / ".venv/bin/vllm"), "serve", str(resolved(cfg["MODEL_PATH"]))]
    for name in shlex.split(cfg["SERVED_MODEL_NAMES"], comments=True):
        command += ["--served-model-name", name]
    if cfg["QUANTIZATION"]:
        command += ["--quantization", cfg["QUANTIZATION"]]
    command += [
               "--host", cfg["HOST"], "--port", cfg["PORT"],
               "--tensor-parallel-size", cfg["TENSOR_PARALLEL_SIZE"],
               "--disable-custom-all-reduce", "--max-model-len", cfg["MAX_MODEL_LEN"],
               "--gpu-memory-utilization", cfg["GPU_MEMORY_UTILIZATION"],
               "--max-num-seqs", cfg["MAX_NUM_SEQS"],
               "--max-num-batched-tokens", cfg["MAX_NUM_BATCHED_TOKENS"],
               "--long-prefill-token-threshold", cfg["LONG_PREFILL_TOKEN_THRESHOLD"],
               "--dtype", "float16", "--mamba-ssm-cache-dtype", "float32",
               "--kv-cache-dtype", "fp8_e4m3", "--attention-config", '{"backend":"FLASHINFER"}',
               "--gdn-prefill-backend", "flashqla_sm75", "--enable-prefix-caching",
               "--async-scheduling", "--reasoning-parser", "qwen3",
               "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
               "--generation-config", "vllm"]
    mtp = int(cfg["MTP_TOKENS"])
    # 小 query Graph 适配器（compat/sm75_fa2_graph.py）支持 B1–B4 连续捕获；
    # 捕获逐步 = (MTP+1) × 请求数，与目标机实机参数一致。
    sizes = [1] if mtp == 0 else [(mtp + 1) * batch for batch in range(1, 5)]
    graph = {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": sizes}
    command += ["--compilation-config", json.dumps(graph, separators=(",", ":"))]
    if mtp:
        command += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": mtp}),
                    "--scheduler-cls", "scheduler_sm75_fair031.SM75FairScheduler031"]
    return command


def runtime_env(cfg):
    env = dict(os.environ)
    for key in ("PYTHONHOME",):
        env.pop(key, None)
    cache = ROOT / ".runtime/cache"
    cuda = resolved(cfg["CUDA_HOME"])
    env.update({
        "PATH": f"{ROOT / '.venv/bin'}:{cuda / 'bin'}:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{cuda / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}",
        "CUDA_HOME": str(cuda), "CC": cfg["CC"], "CXX": cfg["CXX"],
        # 项目内本地适配目录（scheduler_sm75_fair*, sm75_fa2_graph），覆盖上游同名模块。
        "PYTHONPATH": str(ROOT / "compat"), "SM75_FA2_SMALLQ_GRAPH": "1",
        "CUDA_VISIBLE_DEVICES": cfg["GPU_IDS"], "TORCH_CUDA_ARCH_LIST": "7.5",
        "VLLM_API_KEY": cfg["VLLM_API_KEY"], "PYTHONUNBUFFERED": "1",
        "VLLM_CACHE_ROOT": str(cache / "vllm"), "FLASHINFER_CACHE_DIR": str(cache / "flashinfer"),
        "TRITON_CACHE_DIR": str(cache / "triton"), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "FLASH_QLA_SM75_PREBUILT_EXTENSION_PATH": str(ROOT / ".runtime/extensions/flash_qla_sm75_gdn.so"),
        "FLASH_QLA_SM75_ALLOW_JIT": "0", "VLLM_FIREFLY_DIRECT": cfg["FIREFLY_DIRECT"],
        "VLLM_FIREFLY_AR": cfg["FIREFLY_AR"], "VLLM_FIREFLY_AR_BACKEND": cfg["FIREFLY_AR_BACKEND"],
        "VLLM_SPEC_DECODE": "1" if int(cfg["MTP_TOKENS"]) else "0",
        "VLLM_ALLREDUCE_USE_FLASHINFER": "0", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
        "VLLM_PLE_CPU_OFFLOAD": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_MONITOR": "0", "VLLM_TEST_INDEX": "0", "OMP_NUM_THREADS": "2",
        "MAX_JOBS": cfg["MAX_JOBS"], "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1800", "SM75_PROJECT_ROOT": str(ROOT),
    })
    return env


def health_url(cfg):
    host = cfg["HOST"]
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1" if host == "0.0.0.0" else "::1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{cfg['PORT']}"
