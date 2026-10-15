"""SM75 长 context QSA prefill 快路径 (fp16 主 KV, selected-paged fa2)。

沿用 indexer 选出的 topk 集合 (选块顺序不变), 只改 attention 执行: 把原生 KV
cache 重解释成 4-token 子页, 让每个 query 变成一个独立请求, 其页表就是该 query
选中的 512 压缩组 + 因果尾。这样删掉未选中 token 的 dense masked 空转, 长 ctx
下比 triton sparse kernel 明显快 (删掉 O(N) 分块的 masked 空转)。
FlashInfer BatchPrefill (fa2) 内部 fp16 算, sm75 可编译 (use_fp16_qk_reduction=False)。
仅单请求 prefill, 非混合批/投机验证。K/V 视图留在原存储上 (as_strided, 无拷贝)。
"""
import torch
from vllm.triton_utils import tl, triton
from vllm.models.qwen4_exp.nvidia.ops.qsa_fast_short_prefill import _gate


@triton.jit
def _selected_subpages(Indices, Table, Indptr, Pages,
                       SI: tl.constexpr, PAGE: tl.constexpr, WIDTH: tl.constexpr):
    # 每行把选中的 token 映射到 4-token 子页下标, 写进 FlashInfer paged KV 页表。
    row = tl.program_id(0)
    count = tl.load(Indices + row * SI + WIDTH)
    first = tl.load(Indptr + row)
    groups = tl.arange(0, 1024)
    col = groups * 4
    valid = col < count
    token = tl.load(Indices + row * SI + col, mask=valid, other=0)
    physical_page = tl.load(Table + token // PAGE, mask=valid, other=0)
    physical_subpage = physical_page * (PAGE // 4) + (token % PAGE) // 4
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
        # 构造函数会临时建一个 8 MiB int workspace, 立即经公开 API 释放,
        # 让 CUDA 分配器可回收复用给后续模型激活。
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
        # >=512 个完整组时, 页表长度只依赖 tail mod4; 避免每个 QSA 层/长 ctx
        # 分块都重做 CPU plan。
        key = (rows, "full", first % 4) if first >= 2047 else (rows, "short", seq)
        if key == self.plan_key:
            return
        if self.plan_ready is not None:
            # 保护 FlashInfer 可复用 pinned plan 缓冲, 避免上一次 plan 的 H2D
            # 传输还没完就被覆盖。
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
                            q_data_type=torch.float16, kv_data_type=torch.float16,
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
            # 上面 _prepare 的 event 等待已保护旧 pinned staging 缓冲, 这里再换。
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

    def run(self, q, k, v, indices, table, seq, gate, out):
        assert q.dtype == k.dtype == v.dtype == torch.float16
        assert q.shape[1:] == (3, 256) and k.shape[2:] == (1, 256)
        assert q.is_contiguous() and out.is_contiguous() and gate.is_contiguous()
        assert indices.shape == (q.shape[0], 2052)
        assert indices.dtype == table.dtype == torch.int32
        assert table.shape[0] == 1 and table.stride(1) == 1
        assert k.shape[1] % 4 == 0
        assert k.stride(0) == k.shape[1] * k.stride(1)
        assert v.stride(0) == v.shape[1] * v.stride(1)
        assert torch.cuda.current_stream(self.device).cuda_stream == self.stream
        assert not torch.cuda.is_current_stream_capturing()
        self._prepare(q.shape[0], seq)
        _selected_subpages[(q.shape[0],)](
            indices, table, self.indptr, self.pages,
            SI=indices.stride(0), PAGE=k.shape[1], WIDTH=2051, num_warps=4)
        # 原生页拆成 4-token 子页, 不拷贝、不改变交织 FP16 K/V 存储视图的 offset。
        shape = (k.shape[0] * (k.shape[1] // 4), 4, 1, 256)
        kk = k.as_strided(shape, (4*k.stride(1), k.stride(1), k.stride(2), 1))
        vv = v.as_strided(shape, (4*v.stride(1), v.stride(1), v.stride(2), 1))
        self.wrapper.run(q, (kk, vv), out=out)
        _gate[(triton.cdiv(out.numel(), 256),)](
            out, gate, out, out.numel(), B=256, num_warps=4)
        return out


# 每 rank/设备/流共享一个 workspace, 而非每个 QSA 层各分配一个。
_RUNNERS = {}


def run(q, k, v, indices, table, seq, gate, out):
    """按 1024 行有界分块执行, 选块集合与顺序不变。"""
    device = q.device
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    runner = _RUNNERS.get(key)
    if runner is None:
        runner = SelectedPagesRunner(device)
        _RUNNERS[key] = runner
    rows = q.shape[0]
    for first in range(0, rows, 1024):
        stop = min(first + 1024, rows)
        # 投影输出可能 head 内连续但行 stride 更宽; 有界拷贝保持 FlashInfer 输入契约。
        qq = q[first:stop].contiguous()
        runner.run(qq, k, v, indices[first:stop], table,
                   seq - rows + stop, gate[first:stop], out[first:stop])
    return out


def warmup():
    """启动时预编译, 不在未用的启动流上留 workspace。"""
    device = torch.device("cuda", torch.cuda.current_device())
    rows, seq, page = 128, 16403, 400
    q = torch.zeros((rows, 3, 256), dtype=torch.float16, device=device)
    gate, out = torch.zeros_like(q), torch.empty_like(q)
    kv = torch.zeros((1, 1, page, 512), dtype=torch.float16, device=device)
    k, v = kv.transpose(1, 2).split(256, dim=-1)
    table = torch.zeros((1, triton.cdiv(seq, page)), dtype=torch.int32, device=device)
    indices = torch.full((rows, 2052), -1, dtype=torch.int32, device=device)
    indices[:, :2048] = torch.arange(2048, dtype=torch.int32, device=device)
    positions = torch.arange(seq - rows, seq, dtype=torch.int32, device=device)
    tails = (positions + 1) % 4
    for lane in range(3):
        indices[:, 2048+lane] = torch.where(
            tails > lane, ((positions + 1)//4)*4 + lane, -1)
    indices[:, -1] = 2048 + tails
    runner = SelectedPagesRunner(device)
    runner.run(q, k, v, indices, table, seq, gate, out)
    torch.cuda.synchronize(device)
