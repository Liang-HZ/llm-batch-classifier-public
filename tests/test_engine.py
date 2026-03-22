from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from llm_classifier.rate_limiter import (
    SlidingWindowRateLimiter,
    CycleRateLimiter,
    _format_duration,
)


def test_sliding_window_rate_limiter_caps_burst_without_real_api():
    async def scenario():
        limiter = SlidingWindowRateLimiter(
            max_requests=2,
            max_tokens=10_000,
            tokens_per_request=100,
            window_seconds=0.05,
        )
        start = time.monotonic()
        acquired = []

        async def worker():
            await limiter.acquire()
            acquired.append(time.monotonic() - start)

        await asyncio.gather(*(worker() for _ in range(5)))
        return sorted(acquired)

    acquired = asyncio.run(scenario())
    assert acquired[2] >= 0.04
    assert acquired[4] >= 0.09


def test_cycle_rate_limiter_limits_reservations_per_cycle():
    async def scenario():
        limiter = CycleRateLimiter(cycle_seconds=0.05, max_success=2)
        start = time.monotonic()
        acquired = []

        async def worker():
            reservation = await limiter.acquire()
            acquired.append((reservation.cycle_number, time.monotonic() - start))
            await asyncio.sleep(0)
            await reservation.success()

        await asyncio.gather(*(worker() for _ in range(5)))
        limiter.close()
        return acquired

    acquired = asyncio.run(scenario())
    cycle_counts = {}
    for cycle_number, _ in acquired:
        cycle_counts[cycle_number] = cycle_counts.get(cycle_number, 0) + 1

    assert max(cycle_counts.values()) <= 2
    assert len(cycle_counts) >= 3


def test_cycle_rate_limiter_recovers_current_cycle_usage_after_restart(tmp_path: Path):
    db_path = tmp_path / "cycle_rate_limit.sqlite3"

    async def scenario():
        limiter1 = CycleRateLimiter(
            cycle_seconds=10.0,
            max_success=2,
            db_path=db_path,
        )
        res1 = await limiter1.acquire()
        await res1.success()
        await limiter1.acquire()
        limiter1.close()

        limiter2 = CycleRateLimiter(
            cycle_seconds=10.0,
            max_success=2,
            db_path=db_path,
        )
        try:
            await asyncio.wait_for(limiter2.acquire(), timeout=0.05)
            blocked = False
        except asyncio.TimeoutError:
            blocked = True

        state = (limiter2.success_count, limiter2.reserved_count)
        limiter2.close()
        return blocked, state

    blocked, state = asyncio.run(scenario())
    assert blocked is True
    assert state == (1, 1)


def test_bug4_impossible_token_limit_raises_immediately():
    """BUG #4: tokens_per_call > max_tokens 时，SlidingWindowRateLimiter 应立即报错而非挂死。"""
    with pytest.raises(ValueError, match="超过"):
        SlidingWindowRateLimiter(
            max_requests=100,
            max_tokens=50,
            tokens_per_request=100,
        )

    # max_tokens=0 不应触发（0 表示不限）
    limiter = SlidingWindowRateLimiter(
        max_requests=100,
        max_tokens=0,
        tokens_per_request=100,
    )
    assert limiter.max_tokens == 0


def test_format_duration():
    """_format_duration 应自适应选择时/分/秒单位。"""
    assert _format_duration(18000) == "5.0h"
    assert _format_duration(3600) == "1.0h"
    assert _format_duration(90) == "1.5min"
    assert _format_duration(60) == "1.0min"
    assert _format_duration(30) == "30.0s"
    assert _format_duration(0.5) == "0.5s"


def test_rps_window_calculation():
    """RPS * window_seconds 应正确换算为窗口内请求数。"""
    # RPS=3, window=60 → max_requests_in_window=180
    rps, window = 3.0, 60.0
    max_req = int(rps * window) if rps > 0 else 0
    assert max_req == 180

    # TPS=20000, window=60 → max_tokens_in_window=1200000
    tps = 20000.0
    max_tok = int(tps * window) if tps > 0 else 0
    assert max_tok == 1200000

    # RPS=0 → 不限制
    max_req_zero = int(0 * window) if 0 > 0 else 0
    assert max_req_zero == 0


def test_semaphore_does_not_bottleneck_when_window_limits_rate():
    """回归测试：当 Semaphore 足够大时，滑动窗口才是实际的速率瓶颈，Semaphore 不应限制吞吐量。

    模拟高延迟场景（每请求 0.3s），验证：
    - 小窗口 (window=0.1s, max_req=2) 限制了速率
    - 大 Semaphore (20) 不成为瓶颈
    - 在 1 秒内完成的请求数接近 2/0.1s = 20，而不是被 Semaphore 卡住
    """
    async def scenario():
        limiter = SlidingWindowRateLimiter(
            max_requests=2,
            max_tokens=0,
            tokens_per_request=0,
            window_seconds=0.1,
        )
        semaphore = asyncio.Semaphore(20)  # 足够大，不应是瓶颈
        completed = []
        start = time.monotonic()

        async def worker(i: int):
            await limiter.acquire()
            async with semaphore:
                await asyncio.sleep(0.3)  # 模拟高延迟
                completed.append((i, time.monotonic() - start))

        # 启动 10 个任务
        await asyncio.gather(*(worker(i) for i in range(10)))
        return completed

    completed = asyncio.run(scenario())
    assert len(completed) == 10

    # 关键断言：如果 Semaphore 是瓶颈（假设只有 2），10 个任务需要 5*0.3=1.5s
    # 实际上 Semaphore=20 不是瓶颈，滑动窗口每 0.1s 放行 2 个，
    # 10 个任务在 ~0.5s 内全部开始，0.8s 左右全部完成
    total_time = max(t for _, t in completed)
    assert total_time < 1.2  # 宽松上界，确保不被 Semaphore 卡住


def test_shutdown_event_exits_cycle_wait():
    """Graceful shutdown 应中断正在等待周期额度的协程。"""
    async def scenario():
        cycle_limiter = CycleRateLimiter(cycle_seconds=100.0, max_success=1)
        # 占满额度
        res = await cycle_limiter.acquire()
        await res.success()

        shutdown_event = asyncio.Event()

        async def delayed_shutdown():
            await asyncio.sleep(0.05)
            shutdown_event.set()

        asyncio.create_task(delayed_shutdown())
        # 这个 acquire 应该在 shutdown 后立即返回 None（event-driven，无需轮询）
        result = await asyncio.wait_for(
            cycle_limiter.acquire(shutdown_event=shutdown_event),
            timeout=3.0,
        )
        cycle_limiter.close()
        return result

    result = asyncio.run(scenario())
    assert result is None


def test_cycle_reservation_release_frees_slot():
    """release() 应归还当前周期额度，使同一周期内的第二次 acquire 立即成功。"""
    async def scenario():
        limiter = CycleRateLimiter(cycle_seconds=100.0, max_success=1)

        # 占满额度（reserved，尚未 success）
        res1 = await limiter.acquire()
        assert res1 is not None

        # 归还预约 — 不计入 success，释放 reserved slot
        await res1.release()

        # 同一周期内应立即拿到第二次预约
        res2 = await asyncio.wait_for(limiter.acquire(), timeout=0.5)
        assert res2 is not None
        assert res2.cycle_number == res1.cycle_number

        # release() 不应计入 success_count
        assert limiter.success_count == 0

        await res2.release()
        limiter.close()

    asyncio.run(scenario())


def test_sliding_window_shutdown_event_exits_wait():
    """shutdown_event 置位后，SlidingWindowRateLimiter.acquire 应在 ~0.05s 内返回而非阻塞整个窗口。"""
    async def scenario():
        limiter = SlidingWindowRateLimiter(
            max_requests=1,
            max_tokens=0,
            tokens_per_request=0,
            window_seconds=100.0,  # 窗口足够长，正常情况下会长时间阻塞
        )
        # 占满窗口额度
        await limiter.acquire()

        shutdown_event = asyncio.Event()

        async def delayed_shutdown():
            await asyncio.sleep(0.05)
            shutdown_event.set()

        asyncio.create_task(delayed_shutdown())

        start = time.monotonic()
        # 第二次 acquire 应在 shutdown 后返回，而不是等待 100 秒
        result = await asyncio.wait_for(
            limiter.acquire(shutdown_event=shutdown_event),
            timeout=3.0,
        )
        elapsed = time.monotonic() - start

        assert result is None
        assert elapsed >= 0.04   # shutdown 延迟 0.05s，确认没有立即返回
        assert elapsed < 1.0     # 远小于 window_seconds=100，验证 shutdown 生效

    asyncio.run(scenario())
