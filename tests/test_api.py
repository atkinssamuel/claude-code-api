import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.worker_pool import WorkerPool


# -------------------------------------------------------------------------------------
# ---------------------------------- helpers ------------------------------------------
# -------------------------------------------------------------------------------------


def _stream_bytes(*events: dict) -> bytes:
    return b"\n".join(json.dumps(e).encode() for e in events) + b"\n"


def _success_stream(text: str, session_id: str = "sess-1") -> bytes:
    return _stream_bytes(
        {"type": "system", "subtype": "init", "session_id": session_id},
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": text}],
                "model": "claude-sonnet-4-6",
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": text,
            "session_id": session_id,
            "duration_ms": 100,
        },
    )


def _error_stream(error_msg: str, session_id: str = "sess-err") -> bytes:
    return _stream_bytes(
        {"type": "system", "subtype": "init", "session_id": session_id},
        {
            "type": "result",
            "subtype": "error",
            "error": error_msg,
            "session_id": session_id,
        },
    )


class _AsyncLineReader:
    """Async iterator over pre-baked lines, mimics asyncio StreamReader."""

    def __init__(self, data: bytes):
        self._lines = iter(line + b"\n" for line in data.splitlines())

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


def _mock_proc(stdout_data: bytes, exit_code: int = 0, stderr: bytes = b""):
    proc = MagicMock()
    proc.stdout = _AsyncLineReader(stdout_data)
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr)
    proc.wait = AsyncMock(return_value=exit_code)
    return proc


# -------------------------------------------------------------------------------------
# ---------------------------------- fixtures -----------------------------------------
# -------------------------------------------------------------------------------------


_TEST_POOL_SIZE = 4


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture()
async def client():
    from api.main import app, get_optional_pool, get_pool

    pool = WorkerPool(size=_TEST_POOL_SIZE)
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_optional_pool] = lambda: pool

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c

    app.dependency_overrides.clear()


# -------------------------------------------------------------------------------------
# ----------------------------------- tests -------------------------------------------
# -------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["workers"]["idle"] + body["workers"]["busy"] == _TEST_POOL_SIZE


@pytest.mark.anyio
async def test_health_worker_counts(client):
    """Health reflects busy/idle counts from the injected pool."""
    from api.main import app, get_optional_pool, get_pool

    pool = WorkerPool(size=3)
    pool._workers[0].busy = True

    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_optional_pool] = lambda: pool

    resp = await client.get("/health")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["workers"]["idle"] == 2
    assert body["workers"]["busy"] == 1


@pytest.mark.anyio
async def test_query_success(client):
    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_success_stream("Hello, world!")),
    ):
        resp = await client.post(
            "/query", json={"prompt": "say hello", "model": "balanced"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Hello, world!"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["duration_ms"] >= 0


@pytest.mark.anyio
async def test_query_model_tiers(client):
    tier_to_model = {
        "fast": "claude-haiku-4-5-20251001",
        "haiku": "claude-haiku-4-5-20251001",
        "balanced": "claude-sonnet-4-6",
        "sonnet": "claude-sonnet-4-6",
        "best": "claude-opus-4-7",
        "opus": "claude-opus-4-7",
    }

    for tier, expected_model in tier_to_model.items():
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_mock_proc(_success_stream("ok", session_id=f"sess-{tier}")),
        ) as mock_exec:
            resp = await client.post("/query", json={"prompt": "hi", "model": tier})

        assert resp.status_code == 200, f"tier={tier}"
        assert resp.json()["model"] == expected_model, f"tier={tier}"

        call_args = mock_exec.call_args[0]
        assert expected_model in call_args, f"tier={tier} missing model in cmd"


@pytest.mark.anyio
async def test_query_default_model(client):
    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_success_stream("ok")),
    ):
        resp = await client.post("/query", json={"prompt": "hi"})

    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-4-6"


@pytest.mark.anyio
async def test_query_system_prompt(client):
    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_success_stream("ok")),
    ) as mock_exec:
        resp = await client.post(
            "/query",
            json={"prompt": "hi", "system": "You are a pirate."},
        )

    assert resp.status_code == 200
    call_args = mock_exec.call_args[0]
    assert "--system-prompt" in call_args
    idx = list(call_args).index("--system-prompt")
    assert call_args[idx + 1] == "You are a pirate."


@pytest.mark.anyio
async def test_query_error_from_claude(client):
    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_error_stream("model overloaded"), exit_code=1),
    ):
        resp = await client.post("/query", json={"prompt": "hi"})

    assert resp.status_code == 500
    assert "model overloaded" in resp.json()["detail"]


@pytest.mark.anyio
async def test_query_timeout(client):
    async def slow_proc(*args, **kwargs):
        proc = MagicMock()

        async def slow_iter():
            await asyncio.sleep(999)
            return
            yield

        proc.stdout = slow_iter()
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = AsyncMock(return_value=0)
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=slow_proc):
        with patch("api.worker_pool.REQUEST_TIMEOUT_S", 0.05):
            resp = await client.post("/query", json={"prompt": "hi"})

    assert resp.status_code == 504


@pytest.mark.anyio
async def test_pool_exhaustion(client):
    """Requests beyond POOL_SIZE queue and eventually succeed."""
    barrier = asyncio.Event()
    results: list[int] = []

    async def blocked_proc(*args, **kwargs):
        await barrier.wait()
        return _mock_proc(_success_stream("ok"))

    with patch("asyncio.create_subprocess_exec", side_effect=blocked_proc):
        tasks = [
            asyncio.create_task(client.post("/query", json={"prompt": f"q{i}"}))
            for i in range(5)
        ]

        await asyncio.sleep(0.1)

        health = await client.get("/health")
        assert health.json()["workers"]["busy"] == 4

        barrier.set()
        responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
