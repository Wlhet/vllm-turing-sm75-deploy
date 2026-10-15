# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM75 扩展 env —— 对上游 vllm/envs.py 的纯增量, 不再整文件覆盖。

底座保留上游原样 vllm/envs.py; 本模块只装 SM75 自定义的 env getter,
构建时由 install_sm75_overlay.py 把一行 `import vllm.envs_sm75; apply()`
append 到上游 envs.py 尾部触发注入。注入做两件事:

1) 把 EXTENSIONS getter 灌进 vllm.envs.environment_variables —— 上游
   __getattr__ / __dir__ / is_set / validate_environ / enable_envs_cache
   全围绕该 dict, 灌入后 envs.VLLM_FIREFLY_DIRECT 等属性访问自动生效(无需改调用方)。
2) 包一层 vllm.envs.compile_factors, 把若干纯运行时 env(不影响编译图)
   从 hash factors pop 掉 —— 它们只影响运行行为, 不改变编译图。

好处: 上游 envs.py 后续升级随便改(内容级), 本文件只跟「dict + __getattr__
+ compile_factors 返回 dict」这个稳定机制耦合, 冲突面从整文件缩到一处注入。
"""

import os


def _firefly_direct() -> bool:
    """VLLM_FIREFLY_DIRECT 归一化: 默认开 (auto)。firefly 族总开关, 统管:

    - int4(AWQ/GPTQ, W4A16) prefill 现反量化 int8 走 CUTLASS(IMMA)
      (utils/firefly.py, MarlinLinearKernel 混合路径);
    - int8 权重 (compressed-tensors int-quantized) 直接进 firefly int8 GEMM
      (utils/firefly_int8.py, W8A16/W8A8)。

    未设/'auto'/'1'/'on'/'true'/'yes' = 开(合适权重自动启用); '0'/'off'/'false'/
    'no' = 关(回退全上游 marlin)。fp8 权重恒走 marlin, fp8 通信加速走
    VLLM_FIREFLY_AR(独立开关)。纯运行时路径选择, 不改编译图 → pop 出
    compile_factors 复用历史编译产物。
    """
    return os.getenv("VLLM_FIREFLY_DIRECT", "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _firefly_ar_mode() -> bool:
    """VLLM_FIREFLY_AR 归一化: 纯开关, 默认关; '1'/'on'/'true'/'yes' 开。

    fp8 allreduce (FireflyAllReduce, 双 backend P2P/SHM, 见
    VLLM_FIREFLY_AR_BACKEND) 是否启用, 与 firefly 权重路径 (VLLM_FIREFLY_DIRECT)
    无关 (通信独立), fp8 近乎无损 (PLAN-fp8-allreduce §3.5)。09-29 重构
    (3e029be) 曾把 'auto'(跟随总开关) 改成纯开关并默认开, 2026-10-02 应用户
    要求改回默认关 (与重构前 auto→关 的事实行为一致), 显式 =1 才启用。纯
    运行时选择 → pop 出 compile_factors。
    """
    return os.getenv("VLLM_FIREFLY_AR", "0").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _firefly_ar_backend() -> str:
    """VLLM_FIREFLY_AR_BACKEND 归一化: 'auto'(默认, 运行时按 _can_p2p 选 P2P
    优先) / 'p2p'(强制 P2P) / 'shm'(强制 SHM)。

    auto = 有 P2P (NVLink/PCIe 直连) 选 P2P (data/flag 全 device 显存, 无 host
    bounce); 无 P2P (PHB, 如 T10) 回 SHM (/dev/shm + cudaHostRegister)。
    'p2p'/'shm' 强制指定 (P2P 初始化失败仍自动回 SHM 兜底)。
    """
    v = os.getenv("VLLM_FIREFLY_AR_BACKEND", "").strip().lower()
    if v in ("p2p",):
        return "p2p"
    if v in ("shm", "host", "shared"):
        return "shm"
    return "auto"


def _firefly_ar_pipe() -> bool:
    """VLLM_FIREFLY_AR_PIPE 归一化: 默认关; '1'/'on'/'true'/'yes' 开。

    开 = 2-GPU SHM 大消息按 4MB 分块跨 2 stream 全双工 overlap (D2H 发 与 H2D
    收 并发), 全双工 PCIe 链 (二号机 GPU0<->host) 通信项 ~2x, 数值与串行
    firefly_ar_exchange 逐 bit 一致; 半双工链 (T10/一号机) 上分块串行化 ≈ 无
    回退。仅 2-GPU SHM 生效, 小消息 (<=4MB) 仍走串行。默认关保证不改变现有
    行为 (T10 零影响)。见 firefly_allreduce.py / firefly_ar_exchange_pipe。
    """
    return os.getenv("VLLM_FIREFLY_AR_PIPE", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _firefly_tq() -> bool:
    """VLLM_FIREFLY_TQ 归一化: 纯开关, 默认开; '0'/'off'/'false'/'no' 关(回退上游
    原样 FA2/SDPA)。

    开 = TurboQuant KV (tq4/tq8) 的 prefill 走 FlashInfer ragged wrapper
    (O(N) 显存), 替代 FA2 C++ kernel (sm75 必崩: _HAS_FLASH_ATTN 只查 import
    不查 cc) 与 SDPA 回退 (O(N²), 16G 卡长 context OOM)。plan 全在
    TurboQuantMetadataBuilder.build 阶段 (host 侧, cudagraph 外), 数值与
    SDPA/FA2 一致 (flashinfer fa2 kernel)。sm75/T10 上 flashinfer 0.7.0 的
    ragged prefill 已验证 JIT 可用且与 SDPA 对 cos>0.999。
    纯运行时路径选择, 不改编译图 → pop 出 compile_factors 复用历史编译产物;
    0 作 kill-switch 回退 FA2/SDPA, 供其他卡/底座选择。
    见 v1/attention/backends/turboquant_attn.py (install_sm75_overlay 注入) /
    PLAN-firefly-tq-prefill。
    """
    return os.getenv("VLLM_FIREFLY_TQ", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _firefly_defer_recv() -> bool:
    """VLLM_FIREFLY_DEFER 归一化: 纯开关, 默认开; '0'/'off'/'false'/'no' 关。

    开 = PP>1 时非首 stage 的 execute_model 不再在入口阻塞收 pickled tensor
    metadata(那 metadata 只有上一 stage launch 完才发, 把本 rank 的 host 侧输入/
    attention-metadata 准备全卡在"等上一 rank launch"之后), 而是把这次 irecv
    推迟到 model runner 真正要读 intermediate tensors 的那一刻(forward 正前方),
    让 host 准备与上一 stage 的 GPU 计算重叠。设备操作顺序不变(recv→forward→
    send), 数值逐 bit 一致。仅 PP>1 非首 stage 生效; PP1(如当前 TP8 生产)恒
    is_first_rank → 不进该分支 = 完全 no-op, 故默认开对现有生产零影响。
    0 = 旧行为(入口即收), 作回退保险丝。纯 host 调度顺序改动, 不改编译图 →
    pop 出 compile_factors 复用历史编译产物。见 install_sm75_overlay.py 的
    v1/worker/gpu_worker.py 注入(DeferredRecvIntermediateTensors)。
    """
    return os.getenv("VLLM_FIREFLY_DEFER", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _monitor() -> bool:
    """VLLM_MONITOR 归一化: 默认开; '0'/'off'/'false'/'no' 关。

    开 = serve 在 /monitor 挂单文件 HTML 监控页(自拉同源 /metrics 渲染, 无 CDN,
    见 entrypoints/serve/instrumentator/monitor.py); 关 = 不挂该路由。
    """
    return os.getenv("VLLM_MONITOR", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _test_index() -> bool:
    """VLLM_TEST_INDEX 归一化: 默认开; '0'/'off'/'false'/'no' 关。

    开 = serve 在 /test 挂开源测速工具 llm_speedtest 的单文件 HTML 页(前端直连
    模型 API 测 Prefill/Decode 吞吐, 无 CDN, 见
    entrypoints/serve/instrumentator/test.html); 关 = 不挂该路由。
    """
    return os.getenv("VLLM_TEST_INDEX", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _ple_mmap() -> bool:
    """VLLM_PLE_MMAP 归一化: 默认关; '1'/'on'/'true'/'yes' 开。

    开 = qwen4_exp 的 PLE(ngram) 大表 (如 Qwen3.8-Flash-Next 的 105GB 表) 不进
    显存, 走 NVMe mmap 行 gather 经内核 page cache 换页 (见 vllm/vllm_ple_mmap.py)。
    gather 是 CPU 活 + pageable H2D, 改走图外 custom op —— 会改变编译图行为,
    故正常参与 compile_factors hash、也不在 _compile_factors_sm75 pop: 开了就应是
    新编译产物。
    """
    return os.getenv("VLLM_PLE_MMAP", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _ple_mmap_random() -> bool:
    """VLLM_PLE_MMAP_RANDOM 归一化: 默认开; '0'/'off' 关。

    开 = PLE mmap 标 MADV_RANDOM 关内核预读 (稀疏随机行 gather 用不上顺序预读,
    省 NVMe 带宽 + page cache)。见 vllm/vllm_ple_mmap.py::_apply_madv_random。
    """
    return os.getenv("VLLM_PLE_MMAP_RANDOM", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _ple_mmap_prewarm() -> bool:
    """VLLM_PLE_MMAP_PREWARM 归一化: 默认关; '1'/'on' 开。

    开 = 加载时把整表流读一遍填 page cache (无害可驱逐, 取决于空闲内存)。
    见 vllm/vllm_ple_mmap.py::MmapPleTable.prewarm。
    """
    return os.getenv("VLLM_PLE_MMAP_PREWARM", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _ple_mmap_mode() -> str:
    """VLLM_PLE_MMAP_MODE 归一化: PLE(ngram) 表 offload 模式, 三选一可独立设置。

    - disk (默认 / 或仅 VLLM_PLE_MMAP=1): 磁盘 mmap + page cache 换页, 不进显存/
      常驻 RAM。
    - mem: 整表读进 RAM (numpy 连续数组), gather 纯内存无换页 (需 ~95G 主机内存)。
    - vram: 不 offload, 整表进显存 (占 ~95G, 仅小表/验证用)。

    优先级: 显式 MODE > VLLM_PLE_MMAP(1→disk) > 默认 disk。见
    vllm/vllm_ple_mmap.py::mode。纯运行时选择, 不改编译图 → 不进 _compile_factors。
    """
    m = os.getenv("VLLM_PLE_MMAP_MODE", "").strip().lower()
    if m:
        if m in ("mem", "memory", "ram", "cpu"):
            return "mem"
        if m in ("vram", "gpu", "none", "0", "off"):
            return "vram"
        return "disk"
    if os.getenv("VLLM_PLE_MMAP", "").strip().lower() in ("1", "on", "true", "yes"):
        return "disk"
    return "vram"


def _ple_mem_lazy() -> bool:
    """VLLM_PLE_MEM_LAZY 归一化: 默认开; '0'/'off' 关。

    开 = mem 模式走 lazy 后台填表: 启动时只建 header + 空数据区即 attach (秒级,
    不阻塞), 整表由 owner 的后台线程逐片填, gather 按片水位路由、未填片回退磁盘
    读。关 = 旧行为 (owner 阻塞填完整表再 attach, ~11 分钟在启动关键路径), 作回退
    保险丝。纯运行时选择, 不改编译图 → pop 出 compile_factors 复用历史编译产物。
    见 vllm/vllm_ple_mmap.py::MmapPleTable (lazy 分支)。
    """
    return os.getenv("VLLM_PLE_MEM_LAZY", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _ple_mem_fill_workers() -> int:
    """VLLM_PLE_MEM_FILL_WORKERS 归一化: 默认 16。

    lazy-mem 后台填表的并行片数 (NVMe 多队列, 16 足够跑满; 与权重加载并发 IO 想
    少抢可降到 4 —— 反正不阻塞启动)。纯运行时, 不改编译图 → pop 出 compile_factors。
    见 vllm/vllm_ple_mmap.py::MmapPleTable._start_fill。
    """
    try:
        return int(os.environ.get("VLLM_PLE_MEM_FILL_WORKERS", "16"))
    except ValueError:
        return 16


def _ple_host_gather() -> bool:
    """VLLM_PLE_HOST_GATHER 归一化: 默认开; '0'/'off' 关。

    开 = PLE(ngram) 的 ids D2H + 表 gather 从模型 forward 里挪到
    ``Qwen4ExpModelState.prepare_inputs`` (forward 之前), forward 只 return 预填
    buffer 的 view。目的: forward 内不再有 custom op 分裂点 → decode 可走
    FULL_DECODE_ONLY cudagraph (图内无非法 D2H/H2D), 消灭 PIECEWISE 每 step 在
    PLE/GDN/QSA eager 段间隙的 GPU 空转 (参考实现同改造 decode 45→80)。
    关 = 旧行为 (gather 在 forward 内的 custom op 里, 强制 splitting op → 只能
    PIECEWISE), 作回退保险丝。

    本开关**改变 forward 编译图** (op 在/不在 forward 调用路径) → 正常参与编译
    hash, 不 pop 出 compile_factors (开/关各用各的编译产物, 不复用)。
    见 vllm/vllm_ple_mmap.py (host_gather) + install_sm75_overlay.py (model_state 注入)。
    """
    return os.getenv("VLLM_PLE_HOST_GATHER", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _firefly_skillet() -> bool:
    """VLLM_FIREFLY_SKILLET 归一化: 默认关; '1'/'on'/'true'/'yes' 开。

    开 = NVFP4 的 decode(小 M) 走 firefly Skillet (自研 m16n8k8 fp16 tensor-core
    decode GEMM, SM75): 加载期 prepack 把权重 E2M1 nibble 重排成 mma B 操作数
    fragment 顺序 (codes) + 把 E4M3×global 预解成 fp16 per-group scale (scales),
    decode 时权重 E2M1 寄存器现解、乘预解 fp16 scale 直接进 m16n8k8 mma, 消除上游
    Marlin 每 step 的 E4M3 位解码 (那是 NVFP4 比 W4A16 慢 ~10 tok/s 的真因)。

    Skillet 参考了公开的 skinny-GEMV mma 思路但完全自研: m16n8k8 (SM75 高效 tile,
    非老架构 m8n8k4) + 独享快照 + prefill unpack 反解。

    **独享快照**: 开时 process_weights_after_loading 只存 Skillet codes+scales,
    **跳过** marlin int32 repack (省 ~0.5 byte/weight) + **不存** firefly row-major
    快照; prefill(大 M) 用 unpack 把 codes 精确逆解回行主序 (unpack(prepare(w).
    codes)==w 逐 byte), 复用现成 nvfp4_to_int8_cached (c_n 单遍) → int8 → cutlass
    IMMA。每层比基线省 ~0.5 byte/weight, 27B TP4 每卡省 ~3GB+, KV 可涨。

    **改编译图 (开/关选不同 custom op: _ff_nvfp4_skillet_linear vs
    _ff_nvfp4_hybrid_linear), 绝不 pop 出 compile_factors** —— op 选择在 apply_weights
    (被 torch.compile trace) 里, 开/关是不同编译图; pop 会让 on/off 共享 cache hash,
    开时编的图被关时复用 → 访问不存在的 _sk_codes 崩。需形状满足 N%16==0 / K%16==0
    (不满足逐层回退 Marlin)。见 kernels/linear/nvfp4/marlin.py decode 分支 +
    utils/firefly_skillet.py / PLAN-firefly-skillet-decode。
    """
    return os.getenv("VLLM_FIREFLY_SKILLET", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _firefly_skillet_spk() -> int:
    """VLLM_FIREFLY_SKILLET_SPK 归一化: Skillet decode 的 split-K 因子, 默认 8。

    split-K 因子 = gemm block 内 warp 数 (= block 线程数 / 32), 切分 K/16 group。
    SM75 该 kernel 是**延迟受限** (mma 占 ~3%, 带宽占 ~7%), 非带宽 bound: 增大 spk
    → 每 block 独立 mma 链更多, 延迟隐藏更好, m=1 (decode) 提速 (T10 实测 gu
    4352x5120 spk8→spk16 快 30%; dn 1280x17408 快 9%)。取 {2,4,8,16}, 其余值 kernel
    内回退 8。

    与 VLLM_FIREFLY_SKILLET (bool, 开/关选不同 custom op) 同属**改编译图、不 pop**
    一类, 但 spk 更细: 是**传给 custom op 的常量**, 选不同 kernel 模板实例 → 改编译
    图。故**不** pop 出 compile_factors (保留为编译因子), 改 spk 触发重编, 避免旧缓存
    跑出错误 kernel。仅在 SKILLET 开时才有意义 (SKILLET 关走 Marlin, spk 不读)。
    """
    try:
        v = int(os.getenv("VLLM_FIREFLY_SKILLET_SPK", "8").strip())
    except ValueError:
        return 8
    return v if v in (2, 4, 8, 16) else 8


def _firefly_idec_nvfp4() -> bool:
    """VLLM_FIREFLY_IDEC_NVFP4 归一化: 默认关; '1'/'on'/'true'/'yes' 开。

    开 = NVFP4 的 decode(小 M) 不走 Marlin FP4, 改走 firefly int8 decode:
    复用 firefly 快照 (uint8 权重 + fp8 group scale + 预计算 c_n) 现反 int8
    [N,K] 走 cutlass_scaled_mm (SM75 即 IMMA, 2x fp16 TC), 与 prefill 同骨架
    (utils/firefly_nvfp4.py 的 nvfp4_prefill_linear, c_n 已预计算走单遍 cached
    反量化)。NVFP4 Marlin decode 比 AWQ Marlin 重 (S0E5M3 两级 scale 解码交织
    mma 累加), 55529 实测 ~10 tok/s 差距; int8 decode 零新 kernel 零额外显存,
    直接吃 sm75 int8 加速。纯运行时分支选择 (custom op 内按 M 分, 不改编译图)
    → pop 出 compile_factors。需 VLLM_FIREFLY_DIRECT 开 (firefly 快照存在) 才
    生效, 否则该层无快照回退 Marlin。见 kernels/linear/nvfp4/marlin.py decode
    分支 / PLAN-firefly-skillet-decode。
    """
    return os.getenv("VLLM_FIREFLY_IDEC_NVFP4", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _firefly_hc() -> bool:
    """VLLM_FIREFLY_HC 归一化: 默认关; '1'/'on'/'true'/'yes' 开。

    开 = qwen4_exp HyperConnection 三投影 (down+inject merged / down / up) 在单流
    decode(M==1) 走融合标量 GEMV triton kernel (sm75, 不用 tl.dot), 砍 cuBLAS skinny
    GEMM 的 launch/同步固定成本 (Minachist decode-04 hc-fused-int8 同思路)。prefill /
    batch>1 / 非 sm75 / 非 fp16 回退原 cuBLAS。吃 fp16 HC 权重 (vllm_hc_dequant 已
    dequant), 数值与 cuBLAS 同 fp32 累加, 非逐 bit。关 = 完全不改 (上游原路)。
    改变 decode 图内 HC 算子 → 正常参与编译 hash, 不 pop。见 vllm_firefly_hc.py。
    """
    return os.getenv("VLLM_FIREFLY_HC", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


# SM75 自定义 env → getter。install 时灌进 vllm.envs.environment_variables,
# 之后 envs.<NAME> 属性访问 / is_set / validate_environ / __dir__ 自动生效。
EXTENSIONS: dict[str, object] = {
    # firefly(SM75) 族总开关, 默认开(auto, _firefly_direct 归一化):
    #   未设 / auto / 1 = 开(合适权重自动启用): int4(AWQ/GPTQ, W4A16) prefill
    #     现反 int8 走 CUTLASS(IMMA), 小 M/decode 保持 marlin; int8 权重
    #     (W8A16/W8A8) 直接进 firefly int8 GEMM。
    #   0 = 关(全走上游 marlin)。
    # fp8 权重恒走 marlin(sm75 实测 firefly-fp8 不比 marlin 快, 已移除; fp8
    # 通信加速走 VLLM_FIREFLY_AR, 见 PLAN-fp8-allreduce)。纯运行时路径选择,
    # 不改编译图 → pop 出 compile_factors。见 utils/firefly.py / firefly_int8.py。
    "VLLM_FIREFLY_DIRECT": _firefly_direct,
    # 默认 1024: T10 上 p3_perf_sweep(6144x5120 层)测得 int8 反量化 ~1ms/层
    # (M 无关地板), crossover M≈854; M>1024 int8 才稳定快于 marlin(1.13-1.33x)。
    "VLLM_FIREFLY_MIN_M": lambda: int(
        os.environ.get("VLLM_FIREFLY_MIN_M", "1024")
    ),
    # int8 prefill 反量化量化步: "def"(默认) 除法(与两遍 pass2 逐 bit 一致);
    # "fast" 乘倒数(per-row r=1/c_n, 再省 ~30% 反量化, off-by-one ≤0.06%)。
    "VLLM_FIREFLY_DEQUANT_MODEL": lambda: (
        os.environ.get("VLLM_FIREFLY_DEQUANT_MODEL", "def")
    ),
    # fp8 allreduce (FireflyAllReduce) 开关, 默认关(_firefly_ar_mode 归一化,
    # 纯开关); 显式 1 才开。与 firefly 权重路径无关(通信独立, 对任意 TP 都赚)。
    # TP2 每层 2 次 AllReduce 量减半 (fp16->fp8), 省 ~480-530ms/27B prefill。
    # 见 distributed/device_communicators/firefly_allreduce.py / PLAN-fp8-allreduce。
    "VLLM_FIREFLY_AR": _firefly_ar_mode,
    # 只对大消息走 FireflyAllReduce, 小消息回退 NCCL。decode 小消息 (几百 KB)
    # 时 firefly 的 amax 扫描 + 多轮 flag spin 固定开销可能超过砍半省下的传输
    # 时间 → 反而更慢。默认 1MB (fp16 字节): 27B decode M=1 单 AR 8KB << 1MB,
    # 走 NCCL; prefill M>=256 (2MB) 起走 firefly。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_MIN_SIZE": lambda: int(
        os.environ.get("VLLM_FIREFLY_AR_MIN_SIZE", "1048576")
    ),
    # FireflyAllReduce 传输 backend: auto(默认, 运行时 _can_p2p 选 P2P 优先) /
    # p2p(强制) / shm(强制)。P2P 有 NVLink/PCIe 直连时 data/flag 全 device 显存
    # 无 host bounce; 无 P2P (PHB 如 T10) 回 SHM。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_BACKEND": _firefly_ar_backend,
    # SHM 分块全双工流水线开关, 默认关(_firefly_ar_pipe 归一化)。
    # 开 = 2-GPU SHM 大消息按 4MB 分块跨 2 stream overlap (全双工链通信项 ~2x,
    # 数值与串行一致); 半双工链 (T10/一号机) ≈ 无回退。仅 2-GPU SHM 生效。
    "VLLM_FIREFLY_AR_PIPE": _firefly_ar_pipe,
    # tq KV prefill 走 FlashInfer ragged wrapper 开关, 默认开(_firefly_tq 归一化)。
    # 开 = prefill O(N) 显存, 替代 FA2 (sm75 崩) / SDPA (长 ctx OOM); 0 = 回退
    # 上游原样 (FA2/SDPA), 供其他卡/底座选择。
    "VLLM_FIREFLY_TQ": _firefly_tq,
    # 单文件 HTML 监控页开关, 默认开(_monitor 归一化)。
    # 开 = serve 在 /monitor 挂自包含 HTML 看板(纯前端 canvas 图表, 无 CDN,
    # 轮询同源 /metrics); 0/off/false/no = 关(不挂路由)。见
    # entrypoints/serve/instrumentator/monitor.py。
    "VLLM_MONITOR": _monitor,
    # 单文件 HTML 测速页开关, 默认开(_test_index 归一化)。
    # 开 = serve 在 /test 挂 llm_speedtest 测速工具页(开源前端静态页, 直连模型
    # API 测 Prefill/Decode 吞吐, 无 CDN); 0/off/false/no = 关(不挂路由)。见
    # entrypoints/serve/instrumentator/test.html。
    "VLLM_TEST_INDEX": _test_index,
    # PLE(ngram) 大表 NVMe mmap 总开关, 默认关(_ple_mmap 归一化)。
    # 开 = qwen4_exp PLE 表 (如 105GB 的 Qwen3.8-Flash-Next) 不进显存, 走内核
    # page cache 换页行 gather。改变编译图 (gather 走图外 op) → 正常参与编译
    # hash, 不在 _compile_factors_sm75 pop。见 vllm/vllm_ple_mmap.py。
    "VLLM_PLE_MMAP": _ple_mmap,
    # PLE(ngram) 表 offload 模式: disk/mem/vram 三选一 (_ple_mmap_mode 归一化)。
    # disk=磁盘 mmap 换页(默认), mem=整表进 RAM, vram=不进显存 (不 offload, 不打
    # patch → 图不同) → 影响编译图, 正常参与编译 hash, 不 pop。
    "VLLM_PLE_MMAP_MODE": _ple_mmap_mode,
    # PLE mmap 标 MADV_RANDOM 关内核预读, 默认开(_ple_mmap_random 归一化)。
    "VLLM_PLE_MMAP_RANDOM": _ple_mmap_random,
    # PLE mmap 加载时预热 page cache, 默认关(_ple_mmap_prewarm 归一化)。
    "VLLM_PLE_MMAP_PREWARM": _ple_mmap_prewarm,
    # PLE mmap gather 线程数 (运行时直读, 注册仅 validate_environ + hash)。
    "VLLM_PLE_MMAP_WORKERS": lambda: int(
        os.environ.get("VLLM_PLE_MMAP_WORKERS", "32")
    ),
    # PLE mmap 每 gather 任务行数 (运行时直读, 注册仅 validate_environ + hash)。
    "VLLM_PLE_MMAP_CHUNK": lambda: int(
        os.environ.get("VLLM_PLE_MMAP_CHUNK", "2048")
    ),
    # PLE mem 模式 lazy 后台填表开关, 默认开(_ple_mem_lazy 归一化)。
    # 开 = 启动秒级 attach + owner 后台逐片填 + gather 按片水位路由 (未填片回退
    # 磁盘); 0 = 旧行为 (owner 阻塞填完再 attach)。纯运行时, 不改编译图。
    # 见 vllm/vllm_ple_mmap.py::MmapPleTable。
    "VLLM_PLE_MEM_LAZY": _ple_mem_lazy,
    # PLE lazy-mem 后台填表并行片数 (运行时直读, 默认 16; 同权重加载 IO 抢带宽
    # 可降到 4, 反正不阻塞启动)。纯运行时, 不改编译图。
    # 见 vllm/vllm_ple_mmap.py::MmapPleTable._start_fill。
    "VLLM_PLE_MEM_FILL_WORKERS": _ple_mem_fill_workers,
    # PLE host-gather 开关, 默认开(_ple_host_gather 归一化)。
    # 开 = PLE gather 挪进 model_state.prepare_inputs (forward 之前预填常驻 buffer),
    #   forward 只读 buffer view → 无 forward 内 split 点 → decode 可上
    #   FULL_DECODE_ONLY cudagraph; 0 = 旧行为 (gather 在 forward 内 custom op,
    #   必须 PIECEWISE), 作回退保险丝。改变编译图 → 参与编译 hash, 不 pop。
    # 见 vllm/vllm_ple_mmap.py + install_sm75_overlay.py 的 model_state 注入。
    "VLLM_PLE_HOST_GATHER": _ple_host_gather,
    # firefly HC decode 标量 GEMV 开关, 默认关(_firefly_hc 归一化)。
    # 开 = HC 三投影单流 decode 走融合标量 GEMV(sm75, 不用 tl.dot); 改 decode 图内
    # 算子 → 参与编译 hash, 不 pop。见 vllm_firefly_hc.py。
    "VLLM_FIREFLY_HC": _firefly_hc,
    # firefly NVFP4 decode int8 开关, 默认关(_firefly_idec_nvfp4 归一化)。
    # 开 = NVFP4 decode(小 M) 走 firefly int8 (复用快照现反 int8 → cutlass IMMA),
    # 替代 NVFP4 Marlin; 纯运行时分支选择 (custom op 内按 M 分, 不改编译图) →
    # pop 复用既有生产编译产物。需 FIREFLY_DIRECT 开才有快照。见
    # kernels/linear/nvfp4/marlin.py decode 分支。
    "VLLM_FIREFLY_IDEC_NVFP4": _firefly_idec_nvfp4,
    # firefly Skillet NVFP4 decode mma kernel 开关, 默认关(_firefly_skillet 归一化)。
    # 开 = NVFP4 decode(小 M) 走自研 m16n8k8 tensor-core GEMM (prepack 预解 E4M3
    # scale + 独享快照 + prefill unpack 反解), 替代 NVFP4 Marlin decode; 纯运行时
    # 分支选择 (custom op 内按 M 分, 不改编译图) → pop 复用编译产物。需形状
    # N%16==0 / K%16==0。见 kernels/linear/nvfp4/marlin.py + utils/firefly_skillet.py。
    "VLLM_FIREFLY_SKILLET": _firefly_skillet,
    # Skillet decode split-K 因子 (2/4/8/16, 默认 8)。传 custom op 常量 → 选不同
    # kernel 模板 → 改编译图, 故**不** pop 出 compile_factors (改 spk 触发重编)。
    # 仅 SKILLET 开时有意义。见 kernels/linear/nvfp4/marlin.py + firefly_skillet.cu。
    "VLLM_FIREFLY_SKILLET_SPK": _firefly_skillet_spk,
    # firefly PP deferred-recv 开关, 默认开(_firefly_defer_recv 归一化, 纯开关)。
    # 开 = PP>1 非首 stage 把 irecv 从 execute_model 入口推迟到 forward 前, host
    # 准备与上一 stage GPU 计算重叠(设备顺序不变, 逐 bit 一致); 0 = 入口即收(回退)。
    # PP1(当前 TP8)恒 no-op。纯 host 调度顺序, 不改编译图 → pop 复用编译产物。
    # 见 install_sm75_overlay.py 的 v1/worker/gpu_worker.py 注入。
    "VLLM_FIREFLY_DEFER": _firefly_defer_recv,
    # A3(sm75 参考): custom allreduce 在 cuda graph capture 时的图输入策略。
    # auto=full decode 走 registered 快路径, piecewise/prefill 回退 staging
    # buffer(sm75 图私有大 buffer 无法经 CUDA IPC 导出); registered/staging
    # 可强制覆盖。
    "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE": lambda: os.getenv(
        "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE", "auto"
    ),
    # ---- QSA SM75 fast prefill (step2/3/4, 单请求) ----
    # step2 cuBLAS indexer 快路径开关, 默认 0 (fast 版默认 0, entrypoint 显式开 1)。
    # 开 = 单请求 prefill 的 QSA 打分走 cuBLAS fp16 TC (torch.mm) + 原 _topk, 替代
    # triton 逐 query 打分; bf16 存储无损升 fp16 打分。混合/decode 恒走 triton。
    # 纯运行时分支 (custom op 内按 env 分, 不改编译图) → pop 复用编译产物。
    # 见 ops/qsa_fast_indexer.py + indexer_qsa.py 注入。
    "VLLM_QSA_FAST_INDEXER": lambda: os.environ.get(
        "VLLM_QSA_FAST_INDEXER", "0"
    ),
    # step3 短 ctx FA2 快路径开关, 默认 1。开 = 单请求 prefill 且 0<ctx<=16K 走
    # FlashInfer single_prefill (backend=fa2) packed bitmask, 替代 triton sparse
    # attention。fp8 主 KV 内部无损升 fp16。纯运行时 (attention 后端选择, 非
    # linear op) → pop 复用编译产物。见 ops/qsa_fast_short_prefill.py + qsa.py gate。
    "VLLM_QSA_FAST_SHORT_PREFILL": lambda: os.environ.get(
        "VLLM_QSA_FAST_SHORT_PREFILL", "1"
    ),
    # step4 长 ctx FA2 paged 快路径开关, 默认 1。开 = 单请求 prefill 且 16K<ctx
    # <=262144 走 selected-paged FA2 (4-token 子页), 替代 triton sparse attention。
    # 纯运行时 → pop 复用编译产物。见 ops/qsa_fast_long_prefill[_fp8].py + qsa.py gate。
    "VLLM_QSA_FAST_LONG_PREFILL": lambda: os.environ.get(
        "VLLM_QSA_FAST_LONG_PREFILL", "1"
    ),
}


def apply() -> None:
    """把 SM75 扩展 env 注入上游 vllm.envs(幂等)。

    由 install_sm75_overlay.py append 到上游 envs.py 尾部的
    `import vllm.envs_sm75; vllm.envs_sm75.apply()` 触发。此刻 envs 模块的
    environment_variables / compile_factors 均已定义(append 在模块体最末)。
    用 sys.modules 取正在加载的 vllm.envs(即 append 所在模块本身), 不触发
    `import vllm` 的副作用链; 测试里 patch sys.modules["vllm.envs"] 即可复用。
    """
    import sys

    envs = sys.modules["vllm.envs"]

    # 1) 注入 getter —— dict.update 天然幂等(同 key 覆盖同值)。
    envs.environment_variables.update(EXTENSIONS)

    # 2) 包 compile_factors, 从 hash factors pop 掉若干纯运行时 env(幂等: 已包跳过)。
    if not getattr(envs.compile_factors, "_sm75_wrapped", False):
        _orig = envs.compile_factors

        def _compile_factors_sm75():
            factors = _orig()
            # 监控看板/测速页都只挂 HTTP 路由, 不影响编译图; 沿用"关"的历史缓存
            # 签名, 使开/关 UI 都能复用既有生产编译产物。不改 getter: 开关实际仍
            # 按 VLLM_MONITOR / VLLM_TEST_INDEX 原值生效, 这里只归一化 hash 因子:
            # VLLM_MONITOR 沿用历史签名(=False); VLLM_TEST_INDEX 是新增 env, 历史
            # 签名里没有该 key, pop 掉才能与之对齐。
            if "VLLM_MONITOR" in factors:
                factors["VLLM_MONITOR"] = False
            factors.pop("VLLM_TEST_INDEX", None)
            # PLE lazy-mem 两个 env 纯运行时 (后台填表行为 + 填表线程数), 不改编译
            # 图; 都是新增 env, 历史编译签名里没有这俩 key, pop 掉才能与之对齐,
            # 复用既有生产编译产物 (getter 实际仍按 env 原值生效)。
            factors.pop("VLLM_PLE_MEM_LAZY", None)
            factors.pop("VLLM_PLE_MEM_FILL_WORKERS", None)
            # firefly 族 env 分两类, 判据 = 是否改变 apply_weights (被 torch.compile
            # trace) 选哪个 custom op:
            # (a) 不改 op 选择 → 纯运行时, pop 掉复用历史编译产物 (getter 仍按原值生效):
            #     AR (通信 backend) / DEFER (PP recv 时机) / IDEC_NVFP4 (分支在 op 内部,
            #     对 dynamo 是黑盒 leaf 不 trace) / TQ (attention KV, 非 linear op 选择)。
            # (b) **改 op 选择** → 改编译图 → 绝不 pop (开/关各自编图):
            #     DIRECT / SKILLET (开/关走 _ff_nvfp4_hybrid_linear vs
            #     _ff_nvfp4_skillet_linear vs 上游 marlin, 不同 op = 不同图)。pop 会让
            #     on/off 共享同一 cache hash, 开时编的图被关时复用 → 访问不存在的
            #     _sk_codes/_ff_w AttributeError (T10 SKILLET 1→0 实测踩到)。
            # VLLM_FIREFLY 是 DIRECT 取代的旧 key, 历史签名有它, pop 掉避免已设旧 env
            # 时 hash 漂移。
            factors.pop("VLLM_FIREFLY", None)
            factors.pop("VLLM_FIREFLY_AR", None)
            factors.pop("VLLM_FIREFLY_DEFER", None)
            factors.pop("VLLM_FIREFLY_IDEC_NVFP4", None)
            factors.pop("VLLM_FIREFLY_TQ", None)
            # QSA SM75 fast prefill 三开关: 纯运行时分支选择 (custom op 内按 env 分,
            # 不改 apply_weights 选哪个 op, 也不改 attention 编译图) → 都是新增 env,
            # 历史编译签名里没有, pop 掉才能与之对齐, 复用既有生产编译产物。getter
            # 仍按 env 原值生效 (见 EXTENSIONS 注释)。
            factors.pop("VLLM_QSA_FAST_INDEXER", None)
            factors.pop("VLLM_QSA_FAST_SHORT_PREFILL", None)
            factors.pop("VLLM_QSA_FAST_LONG_PREFILL", None)
            # 注意: VLLM_FIREFLY_DIRECT / VLLM_FIREFLY_SKILLET 在 (b) 类, 刻意**不** pop。
            return factors

        _compile_factors_sm75._sm75_wrapped = True  # type: ignore[attr-defined]
        _compile_factors_sm75.__wrapped__ = _orig
        envs.compile_factors = _compile_factors_sm75
