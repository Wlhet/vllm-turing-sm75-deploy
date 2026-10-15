# SPDX-License-Identifier: Apache-2.0
"""仅适配 v0.31 调度队列接口，保留原 B8 算法与 MTP3 条件。"""
from scheduler_sm75_fair import SM75FairScheduler
from vllm.v1.core.sched.scheduler_sm75 import SM75Scheduler


class SM75FairScheduler031(SM75FairScheduler):
    """0.31 把 skipped_waiting 拆成 deferred/持有 KV 的优先队列。"""

    @property
    def skipped_waiting(self):
        # 原 B8 只判断此队列真假；任一新队列非空时保守退回原生调度，
        # 不允许短请求接纳抢过已持有 KV 的请求，也不重新排列任何队列。
        return bool(self.deferred_waiting or self.kv_holding_waiting)

    def schedule(self, throttle_prefills=False):
        # 旧优化仅验证 active==runner slots 的容量模型。0.31 新增独立
        # active 限额，若显式配置为不同值，禁用 B8 而不覆盖上游容量限制。
        if self.max_num_active_reqs != self.max_num_running_reqs:
            return SM75Scheduler.schedule(self, throttle_prefills)
        return super().schedule(throttle_prefills)
