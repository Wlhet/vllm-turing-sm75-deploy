# compat/ — 0.31 运行期本地适配（PYTHONPATH 覆盖，不修改上游）

`scripts/manage.py` 启动时把本目录注入 `PYTHONPATH`（在 site-packages 之前），
只覆盖下面三个文件名；其余模块全部来自安装的官方 vLLM 0.31。

## scheduler_sm75_fair.py

B8 公平调度器（原部署保留件）。仅激活请求数与 runner 槽位一一对应时接管调度，
否则不改动上游行为。依赖上游 overlay 提供的
`vllm.v1.core.sched.scheduler_sm75.SM75Scheduler`。

## scheduler_sm75_fair031.py

`SM75FairScheduler031`：为 vLLM 0.31 新增的 `deferred_waiting` /
`kv_holding_waiting` 队列做保守适配（任一非空时退回原生 SM75 调度，
不重新排列任何队列）；若显式配置了独立的 active 请求限额，同样禁用 B8 而
不覆盖上游容量限制。启用 MTP 时 `manage.py` 以
`--scheduler-cls scheduler_sm75_fair031.SM75FairScheduler031` 选择它。

## sm75_fa2_graph.py

小 query FA2 CUDA Graph 适配器（B1–B4 连续捕获，回退自动 split-KV）。
比 `vendor/vllm-turing/docker/speculative/sm75_fa2_graph.py`（上游 overlay 随
speculative 安装、仅 B1）多一个 `capture_rows_supported` 形状门禁，允许
`cudagraph_capture_sizes = [(MTP+1) × 1..4]`。由 `SM75_FA2_SMALLQ_GRAPH=1`
门控；未捕获到的形状走原动态路径。

这三个文件都经过目标机 4×T10 实机验证（见 `docs/DEPLOYMENT.md`）；本目录
源码不在上游仓库内，更新时请连同 `UPSTREAM.json` 的
`project_local` 哈希一起审查。
