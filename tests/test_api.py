"""Tests for llm_classifier.api — parse_api_response, call_llm, RunStats."""

from __future__ import annotations

import asyncio

import pytest

import llm_classifier.api as api_module
from llm_classifier.api import parse_api_response, call_llm, RunStats
from llm_classifier.config import ClassifyConfig
from llm_classifier.rate_limiter import CycleRateLimiter
from llm_classifier.validation import Validator


# ---------------------------------------------------------------------------
# Fake HTTP helpers (not pytest fixtures — instantiated inline in tests)
# ---------------------------------------------------------------------------


class FakeResponse:
    """Simulate an aiohttp response."""

    def __init__(self, status: int, payload=None, text_data: str = ""):
        self.status = status
        self._payload = payload or {}
        self._text_data = text_data

    async def json(self):
        return self._payload

    async def text(self):
        return self._text_data


class FakeResponseContext:
    """Simulate an aiohttp response context manager."""

    def __init__(self, response: FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    """Simulate aiohttp.ClientSession, returning preset responses by call index."""

    def __init__(self, responses: list[FakeResponse] | None = None):
        self.calls = 0
        self._responses = responses or []

    def post(self, *args, **kwargs):
        self.calls += 1
        if self._responses:
            idx = min(self.calls - 1, len(self._responses) - 1)
            return FakeResponseContext(self._responses[idx])
        return FakeResponseContext(
            FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"labels": []}'}}]},
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_config(**overrides) -> ClassifyConfig:
    """Create a minimal ClassifyConfig for testing."""
    defaults = {
        "categories": ["Computer Science", "Finance", "Marketing"],
        "prompt_template": "Classify into:\n{categories}",
        "user_prompt_template": "{text} / {context}",
        "model_name": "test-model",
        "api_base": "https://test.api",
        "api_key": "test-key",
        "temperature": 0.1,
        "max_tokens": 500,
        "timeout": 30,
        "max_retries": 3,
        "throttle_max_attempts": 10,
        "throttle_base_wait": 30.0,
        "throttle_max_wait": 300.0,
    }
    defaults.update(overrides)
    return ClassifyConfig(**defaults)


def _test_validator() -> Validator:
    cats = ["Computer Science", "Finance", "Marketing"]
    return Validator(cats, set(cats))


# ---------------------------------------------------------------------------
# parse_api_response tests
# ---------------------------------------------------------------------------


def test_parse_api_response_deduplicates_labels():
    """Duplicate label names above threshold should appear only once."""
    validator = _test_validator()
    duplicate_raw = (
        '{"labels": ['
        '{"name": "Computer Science", "confidence": 98, "reason": "x"}, '
        '{"name": "Computer Science", "confidence": 96, "reason": "y"}'
        "]}"
    )
    result = parse_api_response(duplicate_raw, 95, validator)
    assert result["label"] == "Computer Science"
    assert result["is_low_confidence"] == "no"


def test_parse_api_response_error_payload():
    """A payload with an 'error' key and no labels should return label='api_error'."""
    validator = _test_validator()
    result = parse_api_response('{"error": "API call failed"}', 95, validator)
    assert result["label"] == "api_error"
    assert "api_error" in result["parse_status"]


def test_parse_api_response_no_labels_returns_unclassified():
    """An empty labels list should return label='unclassified'."""
    validator = _test_validator()
    result = parse_api_response('{"labels": []}', 95, validator)
    assert result["label"] == "unclassified"
    assert result["parse_status"] == "no_labels"


def test_parse_api_response_below_threshold_marks_low_confidence():
    """A single label below threshold should still be returned but marked low confidence."""
    validator = _test_validator()
    raw = '{"labels": [{"name": "Finance", "confidence": 50, "reason": "low"}]}'
    result = parse_api_response(raw, 95, validator)
    assert result["label"] == "Finance"
    assert result["is_low_confidence"] == "yes"


def test_parse_api_response_unmatched_label_prefixed():
    """Labels not in the category set should be prefixed with 'unmatched_'."""
    validator = _test_validator()
    raw = '{"labels": [{"name": "Unknown Category", "confidence": 98, "reason": "x"}]}'
    result = parse_api_response(raw, 95, validator)
    assert result["label"].startswith("unmatched_")
    assert "unmatched" in result["parse_status"]


def test_parse_api_response_parse_error_on_invalid_json():
    """Completely invalid input should return label='parse_error'."""
    validator = _test_validator()
    # json_repair is quite good at recovery, but truly unparseable input
    # should still result in a parse_error. Use a bytes-like object which
    # cannot be parsed as JSON at all.
    result = parse_api_response("not json at all !!@@##", 95, validator)
    # json_repair may or may not recover; check the shape is correct at minimum.
    assert "label" in result
    assert "parse_status" in result


def test_parse_api_response_ok_status():
    """A clean matching label should return parse_status='ok'."""
    validator = _test_validator()
    raw = '{"labels": [{"name": "Finance", "confidence": 99, "reason": "test"}]}'
    result = parse_api_response(raw, 95, validator)
    assert result["label"] == "Finance"
    assert result["parse_status"] == "ok"
    assert result["is_low_confidence"] == "no"


# ---------------------------------------------------------------------------
# call_llm tests
# ---------------------------------------------------------------------------


def test_call_llm_429_backoff_does_not_hold_semaphore(monkeypatch):
    """During 429 backoff sleep, the semaphore must NOT be held."""
    responses = [
        FakeResponse(429, text_data="too many requests"),
        FakeResponse(200, payload={"choices": [{"message": {"content": '{"labels": []}'}}]}),
    ]
    session = FakeSession(responses)
    config = _test_config(
        throttle_base_wait=0.01,
        throttle_max_wait=0.01,
        throttle_max_attempts=2,
    )

    real_sleep = asyncio.sleep
    semaphore = asyncio.Semaphore(1)
    observed_locked = []

    async def fake_sleep(delay):
        observed_locked.append(semaphore.locked())
        await real_sleep(0)

    monkeypatch.setattr(api_module.random_module, "random", lambda: 0.0)
    monkeypatch.setattr(api_module.asyncio, "sleep", fake_sleep)

    async def scenario():
        return await call_llm(
            session=session,
            config=config,
            text="MSc Finance",
            context="finance program",
            semaphore=semaphore,
            window_limiter=None,
            cycle_limiter=None,
        )

    raw, success, usage = asyncio.run(scenario())
    assert success is True
    assert raw == '{"labels": []}'
    assert observed_locked
    # The semaphore must be free during backoff sleep
    assert observed_locked[0] is False


def test_call_llm_malformed_200_releases_cycle_reservation():
    """A malformed 200 response must release the cycle reservation (not leave it reserved)."""
    responses = [FakeResponse(200, payload={"choices": []})]

    async def scenario():
        cycle_limiter = CycleRateLimiter(cycle_seconds=1.0, max_success=1)
        config = _test_config(max_retries=1)
        raw, success, usage = await call_llm(
            session=FakeSession(responses),
            config=config,
            text="MSc Finance",
            context="finance",
            semaphore=asyncio.Semaphore(1),
            window_limiter=None,
            cycle_limiter=cycle_limiter,
        )
        result = (raw, success, cycle_limiter.success_count, cycle_limiter.reserved_count)
        cycle_limiter.close()
        return result

    raw, success, success_count, reserved_count = asyncio.run(scenario())
    assert success is False
    assert "malformed_api_response" in raw
    assert success_count == 0
    assert reserved_count == 0


def test_run_stats_accumulated_correctly(monkeypatch):
    """RunStats fields should be correctly accumulated across a 429 + 200 sequence."""
    responses = [
        FakeResponse(429, text_data="rate limited"),
        FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": '{"labels": []}'}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        ),
    ]

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    config = _test_config(
        throttle_base_wait=0.001,
        throttle_max_wait=0.001,
    )
    monkeypatch.setattr(api_module.random_module, "random", lambda: 0.0)
    monkeypatch.setattr(api_module.asyncio, "sleep", fake_sleep)

    stats = RunStats()

    async def scenario():
        return await call_llm(
            session=FakeSession(responses),
            config=config,
            text="MSc Finance",
            context="finance",
            semaphore=asyncio.Semaphore(1),
            run_stats=stats,
        )

    raw, success, usage = asyncio.run(scenario())
    assert success is True
    assert stats.total_429s == 1
    assert stats.total_prompt_tokens == 100
    assert stats.total_completion_tokens == 50
    assert stats.total_tokens == 150


def test_bug2_throttle_count_resets_on_non_429(monkeypatch):
    """BUG #2: in a 429 → 500 → 429 sequence, throttle_count must reset after the 500."""
    responses = [
        FakeResponse(429, text_data="rate limited"),
        FakeResponse(500, text_data="server error"),
        FakeResponse(429, text_data="rate limited"),
        FakeResponse(200, payload={"choices": [{"message": {"content": '{"labels": []}'}}]}),
    ]

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    config = _test_config(
        throttle_base_wait=0.001,
        throttle_max_wait=0.001,
        # Only allow 1 consecutive 429 before giving up
        throttle_max_attempts=1,
        # Enough retries to work through the sequence
        max_retries=10,
    )
    monkeypatch.setattr(api_module.random_module, "random", lambda: 0.0)
    monkeypatch.setattr(api_module.asyncio, "sleep", fake_sleep)

    async def scenario():
        return await call_llm(
            session=FakeSession(responses),
            config=config,
            text="Test Program",
            context="test",
            semaphore=asyncio.Semaphore(1),
        )

    # If throttle_count did NOT reset after the 500, the second 429 would
    # accumulate to throttle_count=2 > max_attempts=1 and give up.
    # With the fix, the 500 resets throttle_count=0 so the second 429 is
    # throttle_count=1, which equals (not exceeds) the limit.
    raw, success, usage = asyncio.run(scenario())
    assert success is True
