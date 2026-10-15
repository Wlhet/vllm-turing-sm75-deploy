"""SM75 E4M3FN 主 KV 存储辅助: 位运算解码 + FP16 attention 算术。

量化/cache 写入刻意复用 vLLM 标准 C++ cache op (reshape_and_cache_flash), 本模块
只做读侧解码, 不碰 QSA indexer 的 key/query/logits/选择。
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def decode_e4m3fn(encoded):
    """无 fp8 指令解码全部 256 种 E4M3FN 字节 (sm75 无硬件 fp8)。

    normal (e>=1): 无偏指数 e-7 → fp32 指数 (e-7)+127=e+120, 尾数左移 20 位,
    拼 sign 一次 bitcast 出 fp32; subnormal (e=0): m*2^-9 (m<=7 精确可表示);
    NaN: e=15, m=7; E4M3FN 无 inf, 448.0 饱和即 e=15, m=6。"""
    raw = encoded.to(tl.uint32)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    sign = (raw & 128) << 24
    normal_bits = sign | ((exponent + 120) << 23) | (mantissa << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)
    magnitude = mantissa.to(tl.float32) * 0.001953125
    subnormal_bits = magnitude.to(tl.uint32, bitcast=True) | sign
    subnormal = subnormal_bits.to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, subnormal, normal)
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), value)


@triton.jit(do_not_specialize=["n"])
def _decode_contiguous_kernel(Source, Scale, Output, n, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(Source + offset, mask=offset < n, other=0)
    value = decode_e4m3fn(raw) * tl.load(Scale)
    tl.store(Output + offset, value, mask=offset < n)


def decode_contiguous(source: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 主 KV 字节无损升 fp16 (乘 unit scale), 供 fa2 fp16 计算。"""
    assert source.dtype == torch.uint8 and source.is_contiguous()
    assert scale.numel() == 1 and scale.dtype == torch.float32
    assert scale.device == source.device
    output = torch.empty(source.shape, dtype=torch.float16, device=source.device)
    _decode_contiguous_kernel[(triton.cdiv(source.numel(), 256),)](
        source, scale, output, source.numel(), BLOCK=256, num_warps=4)
    return output


def store_standard(key, value, key_cache, value_cache, slots, k_scale, v_scale):
    """fp8 主 KV 写入, 复用 vLLM 标准 cache op (e4m3, unit scale)。"""
    from vllm._custom_ops import reshape_and_cache_flash
    assert key.dtype == value.dtype == torch.float16
    assert key_cache.dtype == value_cache.dtype == torch.uint8
    assert k_scale.dtype == v_scale.dtype == torch.float32
    assert k_scale.numel() == v_scale.numel() == 1
    reshape_and_cache_flash(key, value, key_cache, value_cache, slots,
                            "fp8_e4m3", k_scale, v_scale)
