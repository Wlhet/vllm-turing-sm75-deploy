# SPDX-License-Identifier: Apache-2.0
"""B8：2K短队头接纳，以及严格三并发下仅在必要时让行的8K服务。"""
import copy
from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler_sm75 import SM75Scheduler

logger = init_logger(__name__)

# 固定筛选边界，不通过环境变量任意放大，以免一次接纳耗尽长请求服务。
SHORT_PREFILL_LIMIT = 2048
MEDIUM_PREFILL_LIMIT = 8192


class SM75FairScheduler(SM75Scheduler):
    """继承真实 AsyncScheduler，保留 MTP/KV/输出占位和取消语义。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 只在短队头获准的一次 schedule 中临时调整私有配置。
        self.scheduler_config = copy.copy(self.scheduler_config)
        self._fair_deferred = set()
        self._fair_last_short_step = -4
        self._fair_medium_turn = 1
        logger.info('SM75 prefill B8: short<=2048, cooldown=4; '
                    'C3 medium<=8192 needs-only rotation, second donor first')

    def _medium_donor(self, throttle_prefills):
        """只针对两长+一个2K至8K队尾，轮流让一条长输入让出当前分块。

        不记住请求对象，不重排队列，取消/抢占后重新按真实RUNNING选择。
        其余队形（已有decode、第四请求、等长输入等）均不使用这条路径。
        """
        if (throttle_prefills or self.skipped_waiting
                or self.num_spec_tokens != 3
                or not self.scheduler_config.enable_chunked_prefill
                or not self.need_mamba_block_aligned_split
                or self.connector is not None or self.ec_connector is not None
                or self.parallel_config.pipeline_parallel_size != 1
                or self.num_waiting_for_streaming_input
                or len(self.running) not in (2, 3)):
            return None
        if len(self.running) == 2:
            if len(self.waiting) != 1 or len(self.running) >= self.max_num_running_reqs:
                return None
            target = next(iter(self.waiting))
            longs = self.running
        else:
            if self.waiting or not self.running[-1].is_prefill_chunk:
                return None
            longs, target = self.running[:2], self.running[-1]
        if (not SHORT_PREFILL_LIMIT < target.num_tokens <= MEDIUM_PREFILL_LIMIT
                or target.has_encoder_inputs or target.num_output_tokens != 0):
            return None
        remaining = target.num_tokens - target.num_computed_tokens
        if remaining <= 0 or any(
                not r.is_prefill_chunk or r.has_encoder_inputs
                or r.num_tokens < 2 * target.num_tokens
                or r.num_tokens - r.num_computed_tokens <= remaining
                for r in longs):
            return None
        # 本轮只验证4096/2048预算；不假定其它页大小也能产生同等服务保证。
        # 一长让行后剩余预算至少2048，真正的Mamba切分仍由父类完成。
        if (self.max_num_scheduled_tokens != 4096
                or self.scheduler_config.long_prefill_token_threshold != 2048
                or not 0 < self.cache_config.block_size <= 2048):
            return None
        # 当中等请求已进入RUNNING，其最后一小块可能本来就能放进余量。
        # 只在两长从页边界开始、远离末尾且尚无输出时证明预算上界：
        # 每长至多 floor(2048 / block_size) * block_size；1600页下余896。
        # 使用包含spec/异步占位的完整demand，不假定剩余prompt就是全部预算。
        block = self.cache_config.block_size
        threshold = self.scheduler_config.long_prefill_token_threshold
        native_spare = self.max_num_scheduled_tokens - 2 * (threshold // block * block)
        if (len(self.running) == 3 and all(
                r.num_output_tokens == 0 and r.num_computed_tokens % block == 0
                and r.num_tokens - r.num_computed_tokens > threshold + 1
                for r in longs)):
            demand = (target.num_tokens_with_spec + target.num_output_placeholders
                      - target.num_computed_tokens)
            if 0 < demand <= native_spare:
                return None
        return longs[self._fair_medium_turn].request_id

    def schedule(self, throttle_prefills=False):
        donor = self._medium_donor(throttle_prefills)
        if donor is not None:
            self._fair_deferred = {donor}
            try:
                result = super().schedule(throttle_prefills)
                # 两长轮流让行：中等请求获得进度时，另一条长请求仍可前进。
                self._fair_medium_turn = 1 - self._fair_medium_turn
                return result
            finally:
                self._fair_deferred = set()
        # 一段连续服务结束后重新从第二条长请求开始；不把上次取消或
        # 奇数分块留下的相位带给下一组用户。无需保存请求对象或ID。
        self._fair_medium_turn = 1
        # 快路径：纯 decode、无等待、全长 prefill 等保持原生批次形状。
        # 不因均分预算改变等长请求完成次序，也不限制普通混合 prefill。
        if (not self.waiting or throttle_prefills or self.skipped_waiting
                or self.num_spec_tokens != 3
                or not self.scheduler_config.enable_chunked_prefill
                or not self.need_mamba_block_aligned_split
                or self.connector is not None or self.ec_connector is not None
                or self.parallel_config.pipeline_parallel_size != 1
                or self.current_step - self._fair_last_short_step < 4
                or len(self.running) + self.num_waiting_for_streaming_input
                    >= self.max_num_running_reqs):
            return super().schedule(throttle_prefills)
        head = next(iter(self.waiting))
        if (head.has_encoder_inputs or head.num_output_tokens != 0
                or not 0 < head.num_tokens <= SHORT_PREFILL_LIMIT
                or any(r.has_encoder_inputs for r in self.running)):
            return super().schedule(throttle_prefills)
        prefills = [r for r in self.running if r.is_prefill_chunk]
        # 只延迟真正的长输入：三条短输入、等长输入等都不走此特殊路径。
        if (not prefills or any(r.num_tokens <= SHORT_PREFILL_LIMIT or
                r.num_tokens - r.num_computed_tokens <= head.num_tokens
                for r in prefills)):
            return super().schedule(throttle_prefills)
        demands = [max(4, r.num_tokens_with_spec + r.num_output_placeholders
                       - r.num_computed_tokens)
                   for r in self.running if not r.is_prefill_chunk]
        total = self.max_num_scheduled_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        fast_total = sum(demands) + head.num_tokens
        if fast_total > total:
            return super().schedule(throttle_prefills)
        # RUNNING 长 prefill 本轮跳过，decode 保留完整 MTP demand，
        # WAITING 队头获得至多2K的预算；真实Mamba checkpoint仍可切分。
        # 1600页下2K初次切分后的448尾块能适配普通批次余量。
        # 不绕过父类对齐，不改请求计数、KV或队列。
        self._fair_deferred = {r.request_id for r in prefills}
        self._fair_last_short_step = self.current_step
        try:
            self.max_num_scheduled_tokens = fast_total
            self.scheduler_config.long_prefill_token_threshold = max(
                head.num_tokens, max(demands, default=4))
            return super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = total
            self.scheduler_config.long_prefill_token_threshold = threshold
            self._fair_deferred = set()

    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                  num_new_local_computed_tokens=0,
                                  num_external_computed_tokens=0):
        """父类 RUNNING 遇到0只跳过本轮，未延迟请求仍走真实对齐实现。"""
        if request.request_id in self._fair_deferred:
            return 0
        return super()._mamba_block_aligned_split(
            request, num_new_tokens, num_new_local_computed_tokens,
            num_external_computed_tokens)
