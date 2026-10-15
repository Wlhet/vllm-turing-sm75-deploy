"""SM75 长 context QSA prefill 快路径 (fp8 E4M3 主 KV, selected-paged fa2)。

与 qsa_fast_long_prefill.py 同构, 但主 KV 是 fp8 (uint8 物理字节)。把原生 cache
重解释成 4-token 子页, 让每个 query 变独立请求, 页表 = 该 query 选中的 512 组 +
因果尾, 删掉未选中 token 的 dense masked 空转。FlashInfer BatchPrefill (fa2)
直接读 fp8 KV, 内部无损升 fp16 算 (sm75 无 fp8 TC, 同一 kernel 下 fp8-KV vs
fp16-KV 输出 max_diff=0, 已验证)。indexer 与选块顺序不变。
仅单请求 prefill。K/V 视图留在原存储 (as_strided, 无拷贝)。
"""
import torch
import logging
import math
from vllm.triton_utils import tl, triton
from vllm.models.qwen4_exp.nvidia.ops.qsa_fast_short_prefill import _gate


@triton.jit
def _selected_subpages(Indices, Table, Indptr, Pages,
                       SI: tl.constexpr, PAGE: tl.constexpr, WIDTH: tl.constexpr,
                       PHYSICAL_BLOCK_SUBPAGES: tl.constexpr):
    # 每行把选中的 token 映射到 4-token 子页下标 (物理块含多个子页)。
    row = tl.program_id(0)
    count = tl.load(Indices + row * SI + WIDTH)
    first = tl.load(Indptr + row)
    groups = tl.arange(0, 1024)
    col = groups * 4
    valid = col < count
    token = tl.load(Indices + row * SI + col, mask=valid, other=0)
    physical_page = tl.load(Table + token // PAGE, mask=valid, other=0)
    physical_subpage = physical_page * PHYSICAL_BLOCK_SUBPAGES + (token % PAGE) // 4
    tl.store(Pages + first + groups, physical_subpage, mask=valid)


class SelectedPagesRunner:
    """单请求 → 每 query 独立稀疏页表; 跨 QSA 层复用同一 runner (共享 workspace)。

    假设 q 行是结束于 seq-1 的连续后缀。K/V 视图留在原存储, 不拷贝。
    """
    def __init__(self, device, disable_split_kv=True):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
        self.device = torch.device(device)
        self.workspace = torch.empty(256,
                                     dtype=torch.uint8, device=self.device)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend="fa2")
        if not callable(getattr(self.wrapper, "workspace_size", None)):
            raise RuntimeError("FlashInfer lacks the required workspace_size API")
        # 构造函数会临时建一个 8 MiB int workspace, 立即经公开 API 释放。
        self.int_workspace = torch.empty(256, dtype=torch.uint8, device=self.device)
        self.wrapper.reset_workspace_buffer(self.workspace, self.int_workspace)
        self.last_workspace_requirements = None
        self.disable_split_kv = disable_split_kv
        self.pages = None
        self.indptr = None
        self.expected_counts = None
        self.plan_key = None
        self.plan_ready = None
        self.stream = torch.cuda.current_stream(self.device).cuda_stream

    def _prepare(self, rows, seq):
        first = seq - rows
        assert first >= 0
        # >=512 个完整组时, 页表长度只依赖 tail mod4; 避免每层/长 ctx 分块重做 plan。
        key = (rows, "full", first % 4) if first >= 2047 else (rows, "short", seq)
        if key == self.plan_key:
            return
        if self.plan_ready is not None:
            # 保护 FlashInfer 可复用 pinned plan 缓冲, 避免上一次 H2D 未完成被覆盖。
            self.plan_ready.synchronize()
        positions = torch.arange(first, seq, dtype=torch.int32)
        counts = torch.clamp((positions + 1) // 4, max=512) * 4 + (positions + 1) % 4
        page_counts = (counts + 3) // 4
        cpu_indptr = torch.zeros(rows + 1, dtype=torch.int32)
        cpu_indptr[1:] = torch.cumsum(page_counts, 0)
        total_pages = int(cpu_indptr[-1])
        if self.pages is None or self.pages.numel() < total_pages:
            self.pages = torch.empty(total_pages, dtype=torch.int32, device=self.device)
        self.indptr = cpu_indptr.to(self.device)
        self.expected_counts = counts
        plan_args = (
            torch.arange(rows + 1, dtype=torch.int32),
            cpu_indptr,
            self.pages[:total_pages],
            (counts - 1) % 4 + 1,
            3, 1, 256, 4)
        plan_options = dict(causal=False, use_fp16_qk_reduction=False,
                            q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn,
                            o_data_type=torch.float16,
                            disable_split_kv=self.disable_split_kv)
        float_need, int_need = self.wrapper.workspace_size(*plan_args, **plan_options)
        float_capacity = max(256, triton.cdiv(int(float_need), 256) * 256)
        int_capacity = max(256, triton.cdiv(int(int_need), 256) * 256)
        changed = False
        if self.workspace.numel() < float_capacity:
            self.workspace = torch.empty(float_capacity, dtype=torch.uint8, device=self.device)
            changed = True
        if self.int_workspace.numel() < int_capacity:
            self.int_workspace = torch.empty(int_capacity, dtype=torch.uint8, device=self.device)
            changed = True
        if changed:
            # 上面 event 等待已保护旧 pinned staging 缓冲, 这里再换。
            self.wrapper.reset_workspace_buffer(self.workspace, self.int_workspace)
        self.last_workspace_requirements = dict(
            rows=rows, seq=seq, float_required_bytes=int(float_need),
            int_required_bytes=int(int_need),
            float_allocated_bytes=self.workspace.numel(),
            int_allocated_bytes=self.int_workspace.numel(),
            selected_page_bytes=self.pages.numel() * self.pages.element_size())
        self.wrapper.plan(*plan_args, **plan_options)
        self.plan_ready = torch.cuda.Event()
        self.plan_ready.record(torch.cuda.current_stream(self.device))
        self.plan_key = key

    def run(self, q, k, v, indices, table, seq, gate, out, *, k_scale=1.0, v_scale=1.0):
        assert q.dtype == torch.float16 and k.dtype == v.dtype == torch.uint8
        assert q.shape[1:] == (3, 256) and k.shape[2:] == (1, 256)
        assert q.is_contiguous() and out.is_contiguous() and gate.is_contiguous()
        assert indices.shape == (q.shape[0], 2052)
        assert indices.dtype == table.dtype == torch.int32
        assert table.shape[0] == 1 and table.stride(1) == 1
        assert k.shape[1] % 4 == 0
        assert k.stride(0) % (4 * k.stride(1)) == 0
        assert v.stride(0) % (4 * v.stride(1)) == 0
        physical_block_subpages = k.stride(0) // (4 * k.stride(1))
        assert physical_block_subpages == v.stride(0) // (4 * v.stride(1))
        assert torch.cuda.current_stream(self.device).cuda_stream == self.stream
        assert not torch.cuda.is_current_stream_capturing()
        self._prepare(q.shape[0], seq)
        _selected_subpages[(q.shape[0],)](
            indices, table, self.indptr, self.pages,
            SI=indices.stride(0), PAGE=k.shape[1], WIDTH=2051,
            PHYSICAL_BLOCK_SUBPAGES=physical_block_subpages, num_warps=4)
        # 原生页拆成 4-token 子页, 不拷贝、不改交织 K/V 存储视图的 offset。
        shape = ((k.shape[0]-1)*physical_block_subpages + k.shape[1]//4, 4, 1, 256)
        kk = k.as_strided(shape, (4*k.stride(1), k.stride(1), k.stride(2), 1))
        vv = v.as_strided(shape, (4*v.stride(1), v.stride(1), v.stride(2), 1))
        self.wrapper.run(q, (kk.view(torch.float8_e4m3fn), vv.view(torch.float8_e4m3fn)),
                         out=out, k_scale=k_scale, v_scale=v_scale)
        _gate[(triton.cdiv(out.numel(), 256),)](
            out, gate, out, out.numel(), B=256, num_warps=4)
        return out


_RUNNERS = {}
_CALLS = 0


def run(q, k, v, indices, table, seq, gate, out, *, k_scale, v_scale):
    """按 1024 行有界分块执行, 直读 fp8 KV, 不建临时 fp16 KV。选块集合不变。

    k_scale/v_scale 必须是 host 标量 (GPU cache scale 的已审计 host 影子),
    拒绝 tensor 以挡住隐式的逐层 GPU 标量同步。0.31 主 KV 未校准 unit, 恒传 1.0。
    """
    global _CALLS
    if not isinstance(k_scale, (float, int)) or not isinstance(v_scale, (float, int)):
        raise TypeError("FP8 selected-page Prefill requires validated host scalar scales")
    if not (math.isfinite(k_scale) and math.isfinite(v_scale) and k_scale > 0 and v_scale > 0):
        raise ValueError("FP8 cache scales must be finite and positive")
    device = q.device
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    runner = _RUNNERS.get(key)
    if runner is None:
        runner = SelectedPagesRunner(device)
        _RUNNERS[key] = runner
        logging.getLogger(__name__).info(
            "SM75 QSA fp8 selected-page Prefill 启用: native_cache_stride=%s, page_size=%d, chunk=1024, fp16_kv_staging=0",
            tuple(k.stride()), k.shape[1])
    rows = q.shape[0]
    for first in range(0, rows, 1024):
        stop = min(first+1024, rows)
        runner.run(q[first:stop].contiguous(), k, v, indices[first:stop], table,
                   seq - rows + stop, gate[first:stop], out[first:stop],
                   k_scale=float(k_scale), v_scale=float(v_scale))
    _CALLS += 1
    return out


def warmup():
    """启动时预编译 fp8/fp16 CUDA 路径, 不在未用启动流上留 cache。"""
    device = torch.device("cuda", torch.cuda.current_device())
    rows, seq, page = 128, 16403, 400
    q = torch.zeros((rows, 3, 256), dtype=torch.float16, device=device)
    gate, out = torch.zeros_like(q), torch.empty_like(q)
    kv = torch.zeros((1, 1, page, 512), dtype=torch.uint8, device=device)
    k, v = kv.transpose(1, 2).split(256, dim=-1)
    table = torch.zeros((1, triton.cdiv(seq, page)), dtype=torch.int32, device=device)
    indices = torch.full((rows, 2052), -1, dtype=torch.int32, device=device)
    indices[:, :2048] = torch.arange(2048, dtype=torch.int32, device=device)
    positions = torch.arange(seq-rows, seq, dtype=torch.int32, device=device)
    tails = (positions+1) % 4
    for lane in range(3):
        indices[:, 2048+lane] = torch.where(tails > lane, ((positions+1)//4)*4+lane, -1)
    indices[:, -1] = 2048+tails
    runner = SelectedPagesRunner(device)
    runner.run(q, k, v, indices, table, seq, gate, out, k_scale=1., v_scale=1.)
    torch.cuda.synchronize(device)
