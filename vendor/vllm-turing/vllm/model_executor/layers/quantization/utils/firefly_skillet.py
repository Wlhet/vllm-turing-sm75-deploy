# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""firefly Skillet: NVFP4 SM75 decode m16n8k8 tensor-core GEMM。

与 firefly_nvfp4.py (NVFP4→int8 prefill) 互补: 这里走 **decode 小 M** 的
m16n8k8 fp16 tensor-core 路径, 权重 E2M1 寄存器现解、乘**预解 fp16 scale**
直接进 mma, 消除上游 Marlin 每 step 的 E4M3 位解码。

NVFP4 数学:
    w_deq[n,k] = E2M1(nibble[n,k]) * E4M3(group_scale[n,k//16]) * global_scale

加载期 prepack (skillet_prepare) 把 E4M3×global 预解成 fp16 per-group scale 并重排
codes 成 m16n8k8 B 操作数 fragment 顺序; decode 时 skillet_gemm 走 m16n8k8 mma。
skillet_unpack 把 codes 精确逆解回行主序 E2M1 nibble, 供 prefill 复用现成
nvfp4_to_int8_cached (unpack(prepare(w).codes) == w 逐 byte)。

独立 torch extension (namespace firefly_skillet), 懒加载, 只编 sm_75。
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# NVFP4 Skillet CUDA kernel (独立 torch extension, 首次用时编译)。
_cuda_mod = None
_cuda_load_attempted = False


def _load_cuda_mod():
    """懒加载 firefly_skillet.cu, 失败返回 None。

    只在首次调用时编译 (缓存到 torch extensions 目录), 不拖慢 vllm import。
    只编 sm_75 (本 overlay 面向 SM75)。
    """
    global _cuda_mod, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_mod
    _cuda_load_attempted = True
    try:
        cu_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "firefly_skillet.cu"
        )
        if not os.path.exists(cu_path):
            logger.warning(
                "firefly_skillet kernel source (firefly_skillet.cu) not found "
                "at %s; NVFP4 Skillet decode 不可用, 回退 Marlin",
                cu_path,
            )
            return None
        from torch.utils.cpp_extension import load as _load_ext

        # 只编 sm_75 (T10)。强制单 arch, 首次 decode JIT 编译 (而非多 arch 默认)。
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        _cuda_mod = _load_ext(
            name="firefly_skillet_cuda",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("firefly_skillet CUDA kernel loaded (NVFP4 decode GEMM)")
    except Exception as e:  # noqa: BLE001 - 编译/无 CUDA 环境回退
        logger.warning("firefly_skillet CUDA ext load failed: %s", e)
        _cuda_mod = None
    return _cuda_mod


def skillet_prepare(
    weight: torch.Tensor,  # uint8 [N, K/2] packed E2M1
    group_scale: torch.Tensor,  # fp8_e4m3 [N, K/16]
    global_scale: float,  # fp32 host 标量
) -> tuple[torch.Tensor, torch.Tensor]:
    """加载期 prepack: (codes, scales)。

    codes  uint8 [N, K/2]  E2M1 nibble 重排成 m16n8k8 B fragment 顺序
    scales fp16  [N, K/16] E4M3×global 预解成 fp16 (RNE round), 核心优化
    要求 N%16==0, K%16==0。
    """
    mod = _load_cuda_mod()
    if mod is None or not weight.is_cuda:
        raise RuntimeError(
            "skillet_prepare 需 CUDA + firefly_skillet kernel 可用; "
            "当前不可用, 请回退 NVFP4 Marlin decode"
        )
    if group_scale.dtype == torch.uint8:
        group_scale = group_scale.view(torch.float8_e4m3fn)
    return tuple(mod.prepare(weight, group_scale, float(global_scale)))


def skillet_unpack(codes: torch.Tensor) -> torch.Tensor:
    """codes → weight_packed uint8[N, K/2] (prepare 重排的精确逆)。

    用途: prefill 解回行主序 E2M1 nibble, 复用现成 nvfp4_to_int8_cached。
    保证 unpack(prepare(w).codes) == w 逐 byte。
    """
    mod = _load_cuda_mod()
    if mod is None or not codes.is_cuda:
        raise RuntimeError(
            "skillet_unpack 需 CUDA + firefly_skillet kernel 可用; "
            "当前不可用, 请回退 NVFP4 Marlin decode"
        )
    return mod.unpack(codes)


def skillet_gemm(
    x: torch.Tensor,  # fp16 [M, K] 行主序
    codes: torch.Tensor,  # uint8 [N, K/2]
    scales: torch.Tensor,  # fp16 [N, K/16]
    n: int,  # 输出列数 N
    spk: int = 8,  # split-K 因子 (block 内 warp 数), 2/4/8/16
) -> torch.Tensor:
    """decode GEMM: out[m,n] = sum_k x[m,k]*w_deq[n,k], 返回 fp16 [M, N]。

    M∈[1,16], N%16==0, K%16==0。内部分配 out 并调 mod.gemm (mutated)。
    spk: split-K 因子, 默认 8。SM75 该 kernel 延迟受限, 增大 spk (如 16) 在
    m=1 提速 (更多独立 mma 链隐藏延迟); 各层最优或不同, 可调。
    """
    mod = _load_cuda_mod()
    if mod is None or not x.is_cuda:
        raise RuntimeError(
            "skillet_gemm 需 CUDA + firefly_skillet kernel 可用; "
            "当前不可用, 请回退 NVFP4 Marlin decode"
        )
    M, K = x.shape
    out = torch.empty((M, n), dtype=torch.float16, device=x.device)
    mod.gemm(out, x, codes, scales, int(spk))
    return out
