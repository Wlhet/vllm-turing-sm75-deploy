# 优化内容与代价

对比基线是官方 vLLM 0.31.0，而不是仓库的任意其他分支。

1. **NVFP4 Firefly 大输入路径**：运行时将权重转换为 INT8 计算形式，利用 SM75 INT8 Tensor Core；小 M 继续走 NVFP4 Marlin。目标是 prefill，不是保证所有 decode 都变快。磁盘权重未变，但额外转换不保证零数值误差。
2. **Firefly All-Reduce**：符合条件的 FP16 中间数据量化成 FP8 通信，TP4 使用蝶形归并，有 P2P 则优先利用；小消息回退 NCCL。主体数据量约减半不等于端到端翻倍，上游未完成该轮量化通信质量验收。
3. **FlashQLA-SM75 GDN prefill**：为相关线性注意力层提供 SM75 计算路径，decode 使用 Triton 路径，不是替换模型所有注意力。
4. **SM75 算子选择与兼容**：例如避免混合量化 FP8 层误选不支持 T10 的 CUTLASS 算子，使用 Marlin FP8；FlashInfer 等路径增加相应适配。
5. **MTP、图捕获与调度适配**：机制来自上游，项目增加 SM75 适配（本基线含 0.31 调度队列保守适配与 B1–B4 小请求 Graph 捕获，见 `compat/`）。当前 MTP3 重复使用同一 MTP 层；FlashInfer 多步草稿融合不可用，会逐步重建 attention metadata。捕获尺寸为 `(MTP+1)×1..4`，配置 FULL_AND_PIECEWISE 不等于所有形状都走 FULL 图。
6. **FP8 KV、前缀缓存、异步调度、分块 prefill**：属于官方已有机制及本部署配置，不应列为项目原创算法。

256K 上下文和 0.92 显存预算是部署参数。监控页、测速页、DFlash、睡眠、CPU offload 不是本项目默认启用的收益。

## 代码位置

以下路径相对于 `vendor/vllm-turing/`：

- `vllm/model_executor/kernels/linear/nvfp4/marlin.py`
- `vllm/model_executor/layers/quantization/utils/firefly_nvfp4.py`
- `vllm/distributed/device_communicators/firefly_allreduce.py`
- `docker/overlay_injections.py`
- `docker/speculative/sm75_fa2_graph.py`（上游 B1 版；运行时由项目根 `compat/sm75_fa2_graph.py` 的 B1–B4 版覆盖）
- `vllm/v1/core/sched/scheduler_sm75.py`（原生 SM75 调度；0.31 公平调度在项目根 `compat/scheduler_sm75_fair031.py`，依赖 `compat/scheduler_sm75_fair.py`）

没有完成受控 A/B，因此本项目不发布端到端加速倍数。后续应分别测 Firefly 计算、通信和 MTP 开关的首 token 延迟、prefill/decode 吞吐、总延迟、显存与草稿接受率，并记录质量变化。
