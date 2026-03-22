from __future__ import annotations


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
