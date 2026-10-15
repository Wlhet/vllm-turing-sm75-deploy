# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""firefly Skillet 数值自测 (自包含, 不依赖 vllm 包)。

直接 `python test_firefly_skillet.py` 跑, 打印 PASS/FAIL。
纯 torch + 直接 import 同目录 firefly_skillet (懒加载 CUDA ext, 只编 sm_75)。

测试:
  1. round-trip (最关键): 随机 uint8 [N, K/2], codes=prepare(w)[0],
     w_rt=unpack(codes), 断言 torch.equal(w_rt, w) 逐 byte。
  2. GEMM 数值: 随机 NVFP4, M=1 和 M=16, 参考=纯 torch 反量化
     (E2M1 LUT × e4m3 × global) cast fp16 后 x@W.T, Skillet=skillet_gemm,
     allclose(out, ref, rtol=2e-2, atol=1e-2), 打印 max abs diff。
  3. 失败打印哪一步 + diff。

说明: 参考用 fp32 反量化 (E2M1 LUT × e4m3 × global) cast fp16。global 取 2 的
幂时, 预解 fp16 scale (E4M3×global) 无舍入 (e4m3 是 fp16 子集), 内核权重与参考
逐元素一致, 仅 fp16 矩阵乘的累积/末位舍入差异。
"""

import os
import sys

import torch

# 直接 import 同目录模块 (不依赖 vllm 包)。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from firefly_skillet import (  # noqa: E402
    skillet_gemm,
    skillet_prepare,
    skillet_unpack,
)


def _e2m1_lut(device) -> torch.Tensor:
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=device, dtype=torch.float32,
    )


def _ref_dequant(weight, group_scale, global_scale, device):
    """纯 torch 参考反量化 → W_deq fp16 [N, K]。

    E2M1 LUT × e4m3 × global, cast fp16。
    """
    N, half_k = weight.shape
    K = half_k * 2
    w = weight.view(torch.uint8)
    lo = w & 0x0F
    hi = (w >> 4) & 0x0F
    lut = _e2m1_lut(device)
    val_lo = lut[(lo & 7).long()] * (1.0 - 2.0 * (((lo >> 3) & 1).to(torch.float32)))
    val_hi = lut[(hi & 7).long()] * (1.0 - 2.0 * (((hi >> 3) & 1).to(torch.float32)))
    deq_e2m1 = torch.stack((val_lo, val_hi), dim=-1).reshape(N, K)
    gs = group_scale.to(torch.float32).repeat_interleave(16, dim=1)
    return (deq_e2m1 * gs * global_scale).to(torch.float16)


def _gen_nvfp4(N, K, seed, device, global_scale):
    """随机 NVFP4 权重 + group scale (合法 e4m3 字节, 避开 NaN 0x7F/0xFF)。"""
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.uint8)
    gs_bytes = torch.randint(0, 127, (N, K // 16), generator=g, dtype=torch.uint8)
    return weight.to(device), gs_bytes.view(torch.float8_e4m3fn).to(device), global_scale


def test_roundtrip(device, N=512, K=2048):
    print(f"[roundtrip] N={N} K={K} ...", flush=True)
    w, gs, gscale = _gen_nvfp4(N, K, seed=1234, device=device, global_scale=0.5)
    codes, scales = skillet_prepare(w, gs, gscale)
    assert codes.dtype == torch.uint8 and codes.shape == (N, K // 2), \
        f"codes 形状/dtype 错: {codes.shape} {codes.dtype}"
    # scales 转置布局 [K/16, N] (prepare 时 .t().contiguous(), decode coalesced 读)
    assert scales.dtype == torch.float16 and scales.shape == (K // 16, N), \
        f"scales 形状/dtype 错: {scales.shape} {scales.dtype}"
    w_rt = skillet_unpack(codes)
    torch.cuda.synchronize()
    if torch.equal(w_rt, w):
        print("[roundtrip] PASS (逐 byte 相等)")
        return True
    diff = (w_rt.view(torch.uint8) != w.view(torch.uint8))
    ndiff = int(diff.sum())
    idx = diff.nonzero()
    print(f"[roundtrip] FAIL: {ndiff} bytes differ, 前 5 处 (n, k/2):")
    for i in range(min(5, ndiff)):
        n, kb = idx[i].tolist()
        print(f"  ({n},{kb}): w={w[n, kb]:#04x} w_rt={w_rt[n, kb]:#04x}")
    return False


def test_gemm(device, N=256, K=1024, global_scale=0.5):
    ok_all = True
    for seed, M in ((7, 1), (7, 16)):
        for spk in (8, 16):  # 验 split-K 因子 (esp spk16 的 threadIdx.x<256 归约 guard)
            w, gs, gscale = _gen_nvfp4(N, K, seed=seed, device=device, global_scale=global_scale)
            codes, scales = skillet_prepare(w, gs, gscale)
            g = torch.Generator(device=device).manual_seed(seed * 31 + 5)
            x = torch.randn(M, K, generator=g, dtype=torch.float16, device=device)
            out = skillet_gemm(x, codes, scales, N, spk)
            torch.cuda.synchronize()

            W_deq = _ref_dequant(w, gs, gscale, device)  # fp16 [N, K]
            # 参考用 fp32 累加再 round fp16 (T10 的 cuBLAS fp16 matmul 是 fp16 累加,
            # 长 K 下不精确, 会假性撑爆容差; kernel 是 fp32 累加更准)。
            ref = (x.float() @ W_deq.float().t()).to(torch.float16)  # fp16 [M, N]

            if out.shape != (M, N):
                print(f"[gemm M={M} spk={spk}] FAIL: out 形状 {out.shape} != {(M, N)}")
                ok_all = False
                continue
            max_diff = float((out.float() - ref.float()).abs().max())
            ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=1e-2)
            status = "PASS" if ok else "FAIL"
            print(f"[gemm M={M} spk={spk}] {status} max_abs_diff={max_diff:.4e}")
            if not ok:
                ok_all = False
            bad = (~torch.isclose(out.float(), ref.float(), rtol=2e-2, atol=1e-2)).nonzero()
            nb = int(bad.shape[0])
            print(f"  {nb} 元素超差, 前 5 处 (m, n): out vs ref")
            for i in range(min(5, nb)):
                m, n = bad[i].tolist()
                print(f"  ({m},{n}): out={out[m, n].item():.4f} ref={ref[m, n].item():.4f}")
    return ok_all


def main():
    if not torch.cuda.is_available():
        print("SKIP: 无 CUDA 设备, 无法跑 NVFP4 Skillet 数值自测 (需 SM75/T10 GPU)")
        return 0
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    ok_rt = test_roundtrip(device)
    ok_gem = test_gemm(device)
    print("=" * 60)
    if ok_rt and ok_gem:
        print("PASS: firefly Skillet 全部自测通过")
        return 0
    print(f"FAIL: roundtrip={'OK' if ok_rt else 'FAIL'} gemm={'OK' if ok_gem else 'FAIL'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
