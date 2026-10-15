# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 overlay: NVFP4 Marlin (W4A16) + firefly(int8 prefill) 混合。

以 v0.30.0 上游 nvfp4/marlin.py 为基线, 叠加 firefly 分支 (VLLM_FIREFLY_DIRECT
门控, off=上游行为):
  - process_weights_after_loading: 先快照原始 NVFP4 布局 (uint8 权重 + fp8 group
    scale + fp32 global scale) + 预计算 per-channel c_n, 再走上游 Marlin repack
    (prepare_fp4_layer_for_marlin 会覆盖 layer.weight 为 Marlin int32, 故须先存)。
  - apply_weights: 大 M (prefill) 把 NVFP4 权重现反量化 int8 走 cutlass_scaled_mm
    (SM75 即 IMMA tensor core, 2x fp16 TC 吞吐); 小 M (decode) 走上游 NVFP4 Marlin。

NVFP4 比 int4 简单: 权重是 uint8 行主序 (2 nibble/byte), 无 Marlin 布局要解析,
反量化 kernel 比 firefly.cu (寄存器现算 marlin 位置) 轻。
见 vllm/model_executor/layers/quantization/utils/firefly_nvfp4.py。
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.firefly_nvfp4 import (
    nvfp4_prefill_linear,
    nvfp4_to_int8,
)
from vllm.model_executor.layers.quantization.utils.firefly_skillet import (
    skillet_gemm,
    skillet_prepare,
    skillet_unpack,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)

import vllm.envs as envs

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


# ---- firefly 混合 op (dynamo/inductor 兼容) ----
# 与 mixed_precision/marlin.py 同款: "门控 + firefly/Marlin 选择 + 执行" 整体包成
# torch.library custom op。firefly 分支判据 (m > min_m, m 是动态 shape) 若留在编译
# 图里, inductor 会把 firefly 路径 (含 custom op) 编进 decode (M=1) 的图, 实测
# decode 性能掉; 图里只留一个不透明 leaf, decode 图干净 (= baseline 性能)。
# op 内部按真实 M 求值: prefill (m>min_m) 现反量化 int8 走 IMMA, decode 走 Marlin。
@torch.library.custom_op("firefly_nvfp4::hybrid_linear", mutates_args=())
def _ff_nvfp4_hybrid_linear(
    x: torch.Tensor,
    ff_weight: torch.Tensor,  # uint8 [N, K/2] 原始 NVFP4
    ff_group_scale: torch.Tensor,  # fp8_e4m3 [N, K/16]
    ff_global_scale: torch.Tensor,  # fp32 [1]
    ff_c_n: torch.Tensor,  # fp32 [N] 预计算 (numel==0 则现算)
    marlin_weight: torch.Tensor,  # Marlin repack int32
    marlin_weight_scale: torch.Tensor,
    marlin_weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor,
    min_m: int,
) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    if m > min_m:
        # prefill: NVFP4 权重现反量化 int8 + cutlass_scaled_mm (SM75 即 IMMA)
        # global_scale 传设备张量 (kernel 读 [0]), 不 .item() (避免 custom op 内 D2H sync)。
        return nvfp4_prefill_linear(
            x,
            ff_weight,
            ff_group_scale,
            ff_global_scale,
            c_n=ff_c_n if ff_c_n.numel() > 0 else None,
        )
    # decode: firefly int8 (VLLM_FIREFLY_IDEC_NVFP4 开 + c_n 已预计算) 走 firefly int8;
    # 否则上游 NVFP4 Marlin (W4A16, E2M1→fp16→fp16 TC)。
    # int8 decode 与 prefill 同骨架 (nvfp4_prefill_linear, c_n 有走单遍 cached 反量化),
    # 零新 kernel 零额外显存, 直接吃 SM75 IMMA。NVFP4 Marlin decode 比 AWQ Marlin 重
    # (S0E5M3 两级 scale 解码交织 mma), int8 是否更快待 55529 实测。
    if envs.VLLM_FIREFLY_IDEC_NVFP4 and ff_c_n.numel() > 0:
        return nvfp4_prefill_linear(
            x,
            ff_weight,
            ff_group_scale,
            ff_global_scale,
            c_n=ff_c_n,
        )
    return apply_fp4_marlin_linear(
        input=x,
        weight=marlin_weight,
        weight_scale=marlin_weight_scale,
        weight_global_scale=marlin_weight_global_scale,
        workspace=workspace,
        size_n=size_n,
        size_k=size_k,
        bias=bias if bias.numel() > 0 else None,
        input_dtype=x.dtype,
    )


@_ff_nvfp4_hybrid_linear.register_fake
def _(
    x, _ff_w, _ff_gs, _ff_gscale, _ff_cn, _m_w, _m_ws, _m_wgs, _ws,
    size_n, _size_k, _bias, _min_m,
):
    # 输出 [*, size_n], dtype = 激活 dtype (与真实路径一致)。
    return torch.empty(x.shape[:-1] + (size_n,), dtype=x.dtype, device=x.device)


# ---- firefly Skillet op (NVFP4 decode m16n8k8 mma, 独享快照) ----
# SKILLET 开时 process_weights 存 **独享快照**: Skillet codes (E2M1 B fragment 序,
# decode GEMM + prefill unpack 反解的源) + 预解 fp16 scales (decode) + 原始 e4m3
# group scale (prefill int8 反量化) + c_n + global scale, 并 **跳过 marlin int32
# repack** (省 ~0.5 byte/weight, 独享快照 vs 基线 marlin+firefly 双份)。
# 本 op 按真实 M 分两路:
#   decode (m<=16): skillet_gemm (m16n8k8 mma, 预解 fp16 scale, E2M1 寄存器现解)
#   prefill (m>16): unpack codes → 行主序 uint8 (unpack(codes)==w 逐 byte) →
#                   现反 int8 (e4m3 scale + c_n 单遍) → cutlass IMMA
# 两路都在 op 黑盒内按 M 求值, 编译图仍是不透明 leaf, 不掉速。
# Skillet 只支持 M∈[1,16]; M>16 全走 int8 prefill (skillet 不覆盖)。
_SKILLET_MAX_M = 16


@torch.library.custom_op("firefly_nvfp4::skillet_linear", mutates_args=())
def _ff_nvfp4_skillet_linear(
    x: torch.Tensor,
    sk_codes: torch.Tensor,  # uint8 [N, K/2] Skillet prepack (E2M1 B fragment 序)
    sk_scales: torch.Tensor,  # fp16 [N, K/16] E4M3×global 预解 (decode 用)
    sk_group_scale: torch.Tensor,  # uint8 (e4m3 字节) [N, K/16] (prefill int8 用)
    sk_c_n: torch.Tensor,  # fp32 [N] per-channel (prefill int8 单遍, numel==0 现算)
    sk_gscale: torch.Tensor,  # fp32 [1] (prefill int8 反量化用)
    bias: torch.Tensor,  # 激活 dtype [N] 或 empty
    size_n: int,
    size_k: int,
    spk: int,  # split-K 因子 (2/4/8/16, 默认 8); 常量, 编译图里是 static
) -> torch.Tensor:
    orig = x.shape
    m = x.numel() // x.shape[-1]
    if m <= _SKILLET_MAX_M:
        # decode: Skillet m16n8k8 mma。skillet_gemm 吃 2D fp16, 出 2D fp16;
        # 多维 x 先 reshape 2D, bf16 cast fp16 (T10 无 bf16 TC, mma 只吃 fp16)。
        x2d = x.reshape(m, -1)
        if x2d.dtype != torch.float16:
            x2d = x2d.to(torch.float16)
        out = skillet_gemm(x2d, sk_codes, sk_scales, size_n, spk)  # [m, N] fp16
        if bias.numel() > 0:
            out = out + bias
        return out.reshape(*orig[:-1], size_n)
    # prefill (m>16): unpack codes → 行主序 uint8 → 现反 int8 (c_n 单遍) → cutlass
    # IMMA。unpack(codes)==w 逐 byte, 复用现成 nvfp4_prefill_linear (返回 [*, N])。
    weight = skillet_unpack(sk_codes)  # uint8 [N, K/2]
    return nvfp4_prefill_linear(
        x,
        weight,
        sk_group_scale,
        sk_gscale,
        c_n=sk_c_n if sk_c_n.numel() > 0 else None,
    )


@_ff_nvfp4_skillet_linear.register_fake
def _(x, _codes, _scales, _gs, _cn, _gscale, _bias, size_n, _size_k, _spk):
    # 输出 [*, size_n], dtype = 激活 dtype。
    return torch.empty(x.shape[:-1] + (size_n,), dtype=x.dtype, device=x.device)


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16) + firefly int8 prefill (SM75)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # SKILLET 独享快照 (开 + N%16==0 + K%64==0): 只存 Skillet codes + 预解
        # fp16 scale + e4m3 group scale + c_n + global, 并跳过 firefly row-major
        # 快照与 marlin int32 repack (每层省 ~0.5 byte/weight, 独享 vs 基线双份)。
        # decode 走 skillet_gemm, prefill 走 unpack(codes)→int8→cutlass。
        # K%64==0 是宽 load 布局所需 (每 chunk 4 组 16B/lane); 不满足该层回退。
        sk_ok = False
        if envs.VLLM_FIREFLY_SKILLET:
            try:
                N = layer.output_size_per_partition
                K = layer.input_size_per_partition
                if N % 16 == 0 and K % 64 == 0:
                    ff_w = layer.weight.detach().view(torch.uint8).contiguous()
                    ff_gs = layer.weight_scale.detach().view(torch.uint8).contiguous()
                    ff_gscale = layer.weight_global_scale.detach().to(torch.float32)
                    gscale_val = float(ff_gscale.item())  # load 期一次, host 标量
                    try:
                        _, cn = nvfp4_to_int8(ff_w, ff_gs, ff_gscale)
                        cn = cn.detach().contiguous()
                    except Exception:  # noqa: BLE001 - c_n 预计算失败, 运行期现算
                        cn = torch.empty(
                            0, dtype=torch.float32, device=ff_w.device
                        )
                    codes, scales = skillet_prepare(
                        ff_w, ff_gs.view(torch.float8_e4m3fn), gscale_val
                    )
                    layer._sk_codes = codes
                    layer._sk_scales = scales
                    layer._sk_group_scale = ff_gs  # e4m3 byte, prefill int8 用
                    layer._sk_c_n = cn
                    layer._sk_gscale = ff_gscale
                    # 独享快照: 原始 weight 已 prepack 进 _sk_codes, 不再需要。释放
                    # layer.weight (0 元素 placeholder, 防别处 AttributeError) 省
                    # ~0.5 byte/weight 换 KV cache; apply_weights SKILLET 分支只用
                    # _sk_*, 不读 layer.weight。
                    # layer.weight 是 nn.Parameter: 直接 `layer.weight = tensor` 会触发
                    # __setattr__ 校验 ("cannot assign ByteTensor as parameter") 抛错;
                    # 改绑 .data (绕过校验, 保留 Parameter 壳) 把存储换成 0 元素张量,
                    # 原 uint8 [N, K/2] 块即被回收。del ff_w 断开最后引用。
                    layer.weight.data = torch.empty(
                        0, dtype=torch.uint8, device=ff_w.device
                    )
                    del ff_w
                    sk_ok = True
                    logger.info_once(
                        "firefly Skillet: 独享快照 (codes + 预解 fp16 scale), 释放原始 weight, 跳 marlin repack"
                    )
            except Exception as e:  # noqa: BLE001 - Skillet 快照失败回退
                logger.warning(
                    "firefly Skillet 快照失败, 该层回退 firefly+Marlin: %s", e
                )
                for attr in (
                    "_sk_codes",
                    "_sk_scales",
                    "_sk_group_scale",
                    "_sk_c_n",
                    "_sk_gscale",
                ):
                    if hasattr(layer, attr):
                        delattr(layer, attr)

        # firefly: 先快照原始 NVFP4 布局 + 预计算 per-channel c_n。
        # prepare_fp4_layer_for_marlin 会把 layer.weight 覆盖成 Marlin int32 布局,
        # 故必须在 repack 之前存下原始 uint8 权重 + fp8 group scale + fp32 global。
        # firefly 快照仅在 SKILLET 未成功时存: SKILLET 成功时 prefill 走 unpack
        # (codes)、decode 走 skillet_gemm (codes), 都不需 row-major 快照 → 跳过省
        # ~0.5 byte/weight (独享)。SKILLET 失败 (shape 不满足/JIT 编译失败) 时
        # sk_ok=False, 这里存 firefly 快照作 prefill+decode-marlin 主路径。
        # 注: Skillet 失败只发生在加载期 (被上方 try/except 捕获), 运行期不会失败。
        if envs.VLLM_FIREFLY_DIRECT and not sk_ok:
            try:
                N = layer.output_size_per_partition
                K = layer.input_size_per_partition
                ff_w = layer.weight.detach().view(torch.uint8).contiguous()  # [N, K/2]
                # fp8_e4m3 → raw byte (bit 重解释, 非数值 cast; .to 会把浮点值 round 成 int 毁字节)
                ff_gs = layer.weight_scale.detach().view(torch.uint8).contiguous()  # [N, K/16]
                ff_gscale = layer.weight_global_scale.detach().to(torch.float32)  # [1]
                layer._ff_w = ff_w
                layer._ff_gs = ff_gs
                layer._ff_gscale = ff_gscale
                # 预计算 c_n (两遍 amax+量化, 仅 load 时一次); 运行期走单遍。
                # 无 GPU/编译失败时 nvfp4_to_int8 回退 PyTorch (慢, 但 load 时一次性)。
                try:
                    _, cn = nvfp4_to_int8(ff_w, ff_gs, ff_gscale)
                    layer._ff_cn = cn.detach().contiguous()
                except Exception:  # noqa: BLE001 - c_n 预计算失败, 运行期现算
                    layer._ff_cn = torch.empty(0, dtype=torch.float32, device=ff_w.device)
                logger.info_once(
                    "firefly NVFP4: 快照原始 FP4 布局 + 预计算 c_n (prefill int8 加速)"
                )
            except Exception as e:  # noqa: BLE001 - 快照失败回退纯 Marlin
                logger.warning(
                    "firefly NVFP4 快照失败, 该层回退纯 Marlin: %s", e
                )
                for attr in ("_ff_w", "_ff_gs", "_ff_gscale", "_ff_cn"):
                    if hasattr(layer, attr):
                        delattr(layer, attr)

        # 上游: Marlin repack (E2M1→int32 + scale 转 S0E5M3 + global scale 校正)。
        # SKILLET 独享快照成功时不走 marlin decode → 跳过 repack 省 int32 权重。
        if not sk_ok:
            prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SKILLET: 独享快照存在 (VLLM_FIREFLY_SKILLET 开) 且激活 fp16/bf16 →
        # 走 Skillet op (decode m<=16 m16n8k8 mma, prefill m>16 unpack→int8→cutlass)。
        if (
            envs.VLLM_FIREFLY_SKILLET
            and hasattr(layer, "_sk_codes")
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            bias_t = (
                bias if bias is not None
                else torch.empty(0, dtype=x.dtype, device=x.device)
            )
            return _ff_nvfp4_skillet_linear(
                x,
                layer._sk_codes,
                layer._sk_scales,
                layer._sk_group_scale,
                layer._sk_c_n,
                layer._sk_gscale,
                bias_t,
                layer.output_size_per_partition,
                layer.input_size_per_partition,
                envs.VLLM_FIREFLY_SKILLET_SPK,
            )
        # firefly 混合: 有快照 (VLLM_FIREFLY_DIRECT 开) 且激活是 fp16/bf16 时,
        # 走 custom op (op 内部按 M 分 prefill-int8 / decode-Marlin)。
        if (
            envs.VLLM_FIREFLY_DIRECT
            and hasattr(layer, "_ff_w")
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            bias_t = (
                bias if bias is not None
                else torch.empty(0, dtype=x.dtype, device=x.device)
            )
            return _ff_nvfp4_hybrid_linear(
                x,
                layer._ff_w,
                layer._ff_gs,
                layer._ff_gscale,
                layer._ff_cn,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                None,  # v0.31.0: per-layer workspace 已移除, 走持久 workspace manager
                layer.output_size_per_partition,
                layer.input_size_per_partition,
                bias_t,
                envs.VLLM_FIREFLY_MIN_M,
            )
        # 非 firefly (off / 无快照 / 非 fp16 激活): 上游 NVFP4 Marlin。
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=None,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
