"""限流子系统 — SlidingWindowRateLimiter / CycleReservation / CycleRateLimiter。

不依赖业务常量或配置模块，限流参数全部通过构造函数传入。
"""

import asyncio
import logging
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .logging_utils import log, add_file_log


def _format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读的时间字符串。"""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds:.1f}s"


# ============================================================
# 滑动窗口限流器
# ============================================================
class SlidingWindowRateLimiter:
    """滑动窗口限流，同时约束请求数和估算 token 数。"""

    def __init__(self, max_requests: int, max_tokens: int, tokens_per_request: int, window_seconds: float = 60.0):
        # BUG #4 FIX: fail-fast if a single request can never fit within the token budget
        if max_tokens > 0 and tokens_per_request > max_tokens:
            raise ValueError(
                f"tokens_per_request ({tokens_per_request}) 超过窗口内 max_tokens ({max_tokens})，"
                f"单次请求永远无法通过 token 限流检查。请调整配置。"
            )
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.tokens_per_request = max(0, tokens_per_request)
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._token_total = 0
        self._lock = asyncio.Lock()

    def _evict_expired(self, now: float):
        while self._events and (now - self._events[0][0]) >= self.window_seconds:
            _, tokens = self._events.popleft()
            self._token_total -= tokens

    async def acquire(self, shutdown_event: asyncio.Event | None = None):
        if self.max_requests <= 0 and self.max_tokens <= 0:
            return

        tokens_needed = self.tokens_per_request
        while True:
            if shutdown_event and shutdown_event.is_set():
                return
            async with self._lock:
                now = time.monotonic()
                self._evict_expired(now)

                request_ok = self.max_requests <= 0 or len(self._events) < self.max_requests
                token_ok = self.max_tokens <= 0 or (self._token_total + tokens_needed) <= self.max_tokens

                if request_ok and token_ok:
                    self._events.append((now, tokens_needed))
                    self._token_total += tokens_needed
                    return

                waits = []
                if self.max_requests > 0 and len(self._events) >= self.max_requests:
                    waits.append(self._events[0][0] + self.window_seconds - now)

                if self.max_tokens > 0 and (self._token_total + tokens_needed) > self.max_tokens:
                    excess_tokens = self._token_total + tokens_needed - self.max_tokens
                    released = 0
                    for ts, event_tokens in self._events:
                        released += event_tokens
                        if released >= excess_tokens:
                            waits.append(ts + self.window_seconds - now)
                            break

                sleep_for = max(0.01, min(waits) if waits else self.window_seconds)

            if shutdown_event:
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=sleep_for)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(sleep_for)


class CycleReservation:
    """一次周期限流预约，最终必须 success() 或 release()。"""

    def __init__(self, limiter: "CycleRateLimiter", cycle_number: int, reservation_id: str):
        self._limiter = limiter
        self.cycle_number = cycle_number
        self.reservation_id = reservation_id
        self._settled = False

    async def success(self):
        if self._settled:
            return
        self._settled = True
        await self._limiter._settle(self.cycle_number, self.reservation_id, success=True)

    async def release(self):
        if self._settled:
            return
        self._settled = True
        await self._limiter._settle(self.cycle_number, self.reservation_id, success=False)


# ============================================================
# 周期限流器
# ============================================================
class CycleRateLimiter:
    """固定窗口周期限流：每个周期内最多 max_success 次预约成功调用。"""

    def __init__(
        self,
        cycle_seconds: float,
        max_success: int,
        log_dir: Path | None = None,
        db_path: Path | None = None,
    ):
        self.cycle_seconds = cycle_seconds
        self.max_success = max_success
        self.log_dir = log_dir
        self.db_path = db_path
        self.cycle_number = 0
        self.cycle_start: float | None = None
        self._success_count = 0
        self._reserved_count = 0
        self._lock = asyncio.Lock()
        self._logged_waiting = False
        self._cycle_log_handler: logging.FileHandler | None = None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path) if db_path else ":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._setup_db()

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def reserved_count(self) -> int:
        return self._reserved_count

    def _setup_db(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cycle_state (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                cycle_start REAL NOT NULL,
                cycle_number INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cycle_reservations (
                reservation_id TEXT PRIMARY KEY,
                cycle_number INTEGER NOT NULL,
                reserved_at REAL NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('reserved', 'success', 'released')),
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cycle_reservations_cycle_status
            ON cycle_reservations (cycle_number, status)
            """
        )
        self._conn.commit()

    def _load_state_row(self):
        return self._conn.execute(
            "SELECT cycle_start, cycle_number FROM cycle_state WHERE id = 1"
        ).fetchone()

    def _save_state(self, cycle_start: float, cycle_number: int):
        self._conn.execute(
            """
            INSERT INTO cycle_state (id, cycle_start, cycle_number)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                cycle_start = excluded.cycle_start,
                cycle_number = excluded.cycle_number
            """,
            (cycle_start, cycle_number),
        )

    def _load_counts(self, cycle_number: int) -> tuple[int, int]:
        success_count = self._conn.execute(
            """
            SELECT COUNT(*) FROM cycle_reservations
            WHERE cycle_number = ? AND status = 'success'
            """,
            (cycle_number,),
        ).fetchone()[0]
        reserved_count = self._conn.execute(
            """
            SELECT COUNT(*) FROM cycle_reservations
            WHERE cycle_number = ? AND status = 'reserved'
            """,
            (cycle_number,),
        ).fetchone()[0]
        return int(success_count), int(reserved_count)

    def _cleanup_old_cycles(self, current_cycle_number: int):
        self._conn.execute(
            "DELETE FROM cycle_reservations WHERE cycle_number < ?",
            (current_cycle_number,),
        )

    def _ensure_cycle_log_handler(self, recovered: bool = False):
        if self._cycle_log_handler or not self.log_dir:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycle_log = self.log_dir / f"cycle_{self.cycle_number}_{ts}.log"
        self._cycle_log_handler = add_file_log(cycle_log)
        if recovered:
            log.info(
                f"=== 周期 {self.cycle_number} 恢复 "
                f"(已成功: {self.success_count}, 已预约: {self.reserved_count}, "
                f"上限: {self.max_success}) ==="
            )

    def _remove_cycle_log_handler(self):
        if self._cycle_log_handler:
            log.removeHandler(self._cycle_log_handler)
            self._cycle_log_handler.close()
            self._cycle_log_handler = None

    def _start_new_cycle(self, now: float):
        self.cycle_start = now
        self.cycle_number += 1
        self._logged_waiting = False
        self._success_count = 0
        self._reserved_count = 0
        self._save_state(self.cycle_start, self.cycle_number)
        self._cleanup_old_cycles(self.cycle_number)
        self._conn.commit()

        self._remove_cycle_log_handler()

        if self.log_dir:
            self._ensure_cycle_log_handler()
            log.info(
                f"=== 周期 {self.cycle_number} 开始 (上限: {self.max_success} 次, "
                f"时长: {_format_duration(self.cycle_seconds)}) ==="
            )

    def _ensure_current_cycle(self, now: float):
        row = self._load_state_row()
        if row is None:
            self.cycle_number = 0
            self._start_new_cycle(now)
            return

        self.cycle_start = float(row["cycle_start"])
        self.cycle_number = int(row["cycle_number"])

        if (now - self.cycle_start) < self.cycle_seconds:
            self._success_count, self._reserved_count = self._load_counts(self.cycle_number)
            self._ensure_cycle_log_handler(recovered=True)
            return

        cycles_elapsed = int((now - self.cycle_start) // self.cycle_seconds)
        if cycles_elapsed <= 0:
            cycles_elapsed = 1
        self.cycle_start += self.cycle_seconds * cycles_elapsed
        self.cycle_number += cycles_elapsed
        self._logged_waiting = False
        self._success_count = 0
        self._reserved_count = 0
        self._save_state(self.cycle_start, self.cycle_number)
        self._cleanup_old_cycles(self.cycle_number)
        self._conn.commit()

        self._remove_cycle_log_handler()

        if self.log_dir:
            self._ensure_cycle_log_handler()
            log.info(
                f"=== 周期 {self.cycle_number} 开始 (上限: {self.max_success} 次, "
                f"时长: {_format_duration(self.cycle_seconds)}) ==="
            )

    async def acquire(self, shutdown_event: asyncio.Event | None = None) -> CycleReservation | None:
        """预约一个周期额度。失败/429 时必须 release()，成功时必须 success()。"""
        if self.max_success <= 0:
            return None

        while True:
            if shutdown_event and shutdown_event.is_set():
                return None

            async with self._lock:
                now = time.time()
                self._ensure_current_cycle(now)
                current_used = self.success_count + self.reserved_count

                if current_used < self.max_success:
                    reservation_id = uuid4().hex
                    self._conn.execute(
                        """
                        INSERT INTO cycle_reservations (
                            reservation_id, cycle_number, reserved_at, status, updated_at
                        ) VALUES (?, ?, ?, 'reserved', ?)
                        """,
                        (reservation_id, self.cycle_number, now, now),
                    )
                    self._conn.commit()
                    self._reserved_count += 1
                    return CycleReservation(self, self.cycle_number, reservation_id)

                remaining = self.cycle_seconds - (now - self.cycle_start)
                if not self._logged_waiting:
                    self._logged_waiting = True
                    log.info(
                        f"周期 {self.cycle_number} 限额已满 "
                        f"({self.success_count}+{self.reserved_count}/{self.max_success}), "
                        f"等待 {_format_duration(remaining)} 后进入下一周期"
                    )

            # shutdown_event.wait() 带超时：正常时睡到周期结束，shutdown 时立即唤醒
            wait_seconds = max(remaining, 0.01)
            if shutdown_event:
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=wait_seconds)
                except asyncio.TimeoutError:
                    pass  # 超时 = 周期到期，继续循环进入新周期
            else:
                await asyncio.sleep(wait_seconds)

    async def _settle(self, cycle_number: int, reservation_id: str, success: bool):
        async with self._lock:
            now = time.time()
            new_status = "success" if success else "released"
            cursor = self._conn.execute(
                """
                UPDATE cycle_reservations
                SET status = ?, updated_at = ?
                WHERE reservation_id = ? AND cycle_number = ? AND status = 'reserved'
                """,
                (new_status, now, reservation_id, cycle_number),
            )
            if cursor.rowcount:
                self._conn.commit()
            self._ensure_current_cycle(now)

    def close(self):
        """清理周期日志 handler。"""
        self._remove_cycle_log_handler()
        self._conn.close()
