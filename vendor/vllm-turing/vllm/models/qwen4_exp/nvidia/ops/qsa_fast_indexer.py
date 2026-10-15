"""SM75 单请求 QSA prefill 打分快路径 (cuBLAS, VLLM_QSA_FAST_INDEXER 门控)。

只改打分实现, 不改变选出的 topk 集合。decode / 混合批 / 长 context 一律保留
原 triton 路径。仅在"无 decode、恰好一个 prefill"的单请求分支调用 (见
indexer_qsa.forward 的接线)。分配全部局部于本次调用, 不跨流/跨请求共享 workspace。
"""
import os

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
_logged = False
# 打分 raw 缓冲上限 (32 MiB), 超限按行分块 (见下方 rows_per_chunk)。
_RAW_SCORE_BYTES = 32 * 1024 * 1024


@triton.jit(do_not_specialize=["columns", "visible_columns", "pages"])
def _gather_keys(cache, table, output, columns, visible_columns,
                 stride_page, stride_token, stride_dim, pages,
                 PAGE_SIZE: tl.constexpr,
                 BLOCK: tl.constexpr):
    x = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    col = x // 128
    dim = x % 128
    live = col < visible_columns
    page = tl.load(table+col//PAGE_SIZE, live, other=0)
    physical_valid = (page >= 0) & (page < pages)
    values = tl.load(cache+page.to(tl.int64)*stride_page
                     +(col % PAGE_SIZE)*stride_token+dim*stride_dim,
                     live & physical_valid, other=float("nan"))
    # padding 位置不读 cache 字节 (可能是 NaN/未初始化/未来页); 越界的 *live*
    # 页保留 NaN 而非掩盖非法 metadata。
    tl.store(output+x, tl.where(live, values, 0.0), col < columns)


@triton.jit(do_not_specialize=["columns"])
def _relu_sum_heads(raw, logits, visible, columns, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
    head = tl.arange(0, 4)
    values = tl.load(raw+(row*4+head[:, None])*columns+col[None, :],
                     col[None, :] < columns, other=0.0)
    score = tl.sum(tl.maximum(values, 0.0), axis=0)
    score = tl.where(col < tl.load(visible+row), score, float("-inf"))
    tl.store(logits+row*columns+col, score, col < columns)


def gather_keys(k_cache, page_table, max_seq_len):
    """gather 出可见压缩块供打分; 不额外做缓存。"""
    visible_columns = max_seq_len // 4
    columns = triton.cdiv(triton.cdiv(max_seq_len, 4), 64)*64
    columns = min(max(64, columns), page_table.shape[1]*k_cache.shape[1])
    keys = torch.empty((columns, 128), dtype=torch.float16, device=k_cache.device)
    _gather_keys[(triton.cdiv(columns*128, 256),)](
        k_cache, page_table, keys, columns, visible_columns,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(3), k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1], BLOCK=256,
        num_warps=4)
    return keys


def try_qsa_select_paged_prefill_cublas(
    q, k_cache, page_table, query_start_loc, visible_blocks,
    token_topk, compress_ratio, max_query_len, block_indices, max_seq_len,
    *, single_request_prefill,
):
    """命中时填 block_indices 并返回 True, 否则返回 False 交给 triton 回退。

    worker 必须从正常单执行线程提交算子。fp16 累加精度策略只在本函数 cuBLAS
    提交期间收紧, finally 恢复, 不做永久性精度策略改动, 也不降 dtype。
    dispatch 之后的错误向上抛出, 不静默转成回退 (由调用方 try/except 兜底)。
    """
    if os.getenv("VLLM_QSA_FAST_INDEXER", "0") != "1":
        return False
    if not single_request_prefill or not current_platform.is_device_capability(75):
        return False
    # 0.31 indexer cache 是 bf16 (无损基线, 保留存储不动); bf16→fp16 无损
    # (7 位尾数装得进 10 位), gather 内升 fp16 存 keys, mm 仍走 fp16 TC。
    if (q.dtype not in (torch.float16, torch.bfloat16)
            or k_cache.dtype not in (torch.float16, torch.bfloat16)
            or q.ndim != 3 or q.shape[1:] != (4, 128)
            or not q.is_contiguous() or q.shape[0] < 128
            or k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128)
            or page_table.ndim != 2 or page_table.shape[0] != 1
            or query_start_loc.shape != (2,)
            or visible_blocks.shape != (q.shape[0],)
            or block_indices.shape != (q.shape[0], 512)
            or token_topk != 2048 or compress_ratio != 4
            or max_query_len < q.shape[0] or not 2048 <= max_seq_len <= 262144
            or max_seq_len//4 > page_table.shape[1]*k_cache.shape[1]):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import _topk
    if q.dtype != torch.float16:
        q = q.to(torch.float16)  # bf16→fp16 无损

    keys = gather_keys(k_cache, page_table, max_seq_len)
    columns = keys.shape[0]
    rows_per_chunk = min(q.shape[0], max(1, _RAW_SCORE_BYTES//(4*columns*4)))
    raw = torch.empty((rows_per_chunk*4, columns), dtype=torch.float32, device=q.device)
    logits = torch.empty((rows_per_chunk, columns), dtype=torch.float32, device=q.device)
    workspace = torch.empty((1024*1024,), dtype=torch.uint8, device=q.device)
    previous_reduction = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    previous_accum = torch.backends.cuda.matmul.allow_fp16_accumulation
    try:
        # 收紧 fp16 累加精度, 与 triton 打分逐位对齐 (无损前提)。
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        for start in range(0, q.shape[0], rows_per_chunk):
            count = min(rows_per_chunk, q.shape[0]-start)
            torch.mm(q[start:start+count].reshape(count*4, 128), keys.T,
                     out_dtype=torch.float32, out=raw[:count*4])
            _relu_sum_heads[(count, triton.cdiv(columns, 256))](
                raw, logits, visible_blocks[start:start+count], columns,
                BLOCK=256, num_warps=4)
            _topk(logits[:count], visible_blocks[start:start+count], 2048, 4,
                  block_indices[start:start+count], workspace)
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous_reduction
        torch.backends.cuda.matmul.allow_fp16_accumulation = previous_accum
    global _logged
    if not _logged:
        logger.info("SM75 QSA cuBLAS indexer 快路径启用: 单请求 prefill, FP16 输入/"
                    "FP32 logits, raw 打分 <=32 MiB, page_size=%d; 混合/decode 走原路径",
                    k_cache.shape[1])
        _logged = True
    return True


def warmup_qsa_cublas_indexer(device, page_size=100):
    """小尺寸真实 launch, 在图外跑; 避免分配 256K workspace。"""
    if os.getenv("VLLM_QSA_FAST_INDEXER", "0") != "1":
        return False
    if not current_platform.is_device_capability(75):
        return False
    context = 4096
    pages = triton.cdiv(context//4, page_size)
    q = torch.ones((128, 4, 128), device=device, dtype=torch.float16)
    cache = torch.ones((pages, page_size, 1, 128), device=device, dtype=torch.float16)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    starts = torch.tensor([0, 128], device=device, dtype=torch.int32)
    visible = (torch.arange(context-128, context, device=device, dtype=torch.int32)+1)//4
    indices = torch.empty((128, 512), device=device, dtype=torch.int32)
    return try_qsa_select_paged_prefill_cublas(
        q, cache, table, starts, visible, 2048, 4, 128, indices, context,
        single_request_prefill=True)
