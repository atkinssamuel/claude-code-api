import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.worker_pool import Worker, WorkerPool


# -------------------------------------------------------------------------------------
# ---------------------------------- helpers ------------------------------------------
# -------------------------------------------------------------------------------------


def _stream_bytes(*events: dict) -> bytes:
    return b"\n".join(json.dumps(e).encode() for e in events) + b"\n"


def _success_stream(text: str, session_id: str = "sess-x") -> bytes:
    return _stream_bytes(
        {"type": "system", "subtype": "init", "session_id": session_id},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "result": text,
            "session_id": session_id,
        },
    )


class _AsyncLineReader:
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
# ----------------------------------- tests -------------------------------------------
# -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_pool_idle_count():
    pool = WorkerPool(size=3)
    assert pool.idle_count == 3
    assert pool.busy_count == 0


@pytest.mark.asyncio
async def test_worker_pool_single_query():
    pool = WorkerPool(size=2)

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_success_stream("hi there")),
    ):
        text, model, duration_ms = await pool.query("say hi", model="balanced")

    assert text == "hi there"
    assert model == "claude-sonnet-4-6"
    assert duration_ms >= 0


@pytest.mark.asyncio
async def test_worker_session_continuity():
    """Second call on the same worker uses --resume with the previous session_id."""
    pool = WorkerPool(size=1)

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_success_stream("first", session_id="sess-abc")),
    ):
        await pool.query("first prompt")

    assert pool._workers[0].session_id == "sess-abc"

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_success_stream("second", session_id="sess-abc")),
    ) as mock_exec:
        await pool.query("second prompt")

    call_args = list(mock_exec.call_args[0])
    assert "--resume" in call_args
    idx = call_args.index("--resume")
    assert call_args[idx + 1] == "sess-abc"


@pytest.mark.asyncio
async def test_session_not_resumed_when_system_prompt():
    """System prompt forces a fresh session (no --resume)."""
    pool = WorkerPool(size=1)
    pool._workers[0].session_id = "old-sess"

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(_success_stream("ok")),
    ) as mock_exec:
        await pool.query("hi", system="You are a pirate.")

    call_args = list(mock_exec.call_args[0])
    assert "--resume" not in call_args


@pytest.mark.asyncio
async def test_worker_pool_concurrent_limit():
    """Exactly POOL_SIZE workers run concurrently; extras queue."""
    pool = WorkerPool(size=2)
    barrier = asyncio.Event()
    started: list[int] = []

    async def slow_proc(*args, **kwargs):
        started.append(1)
        await barrier.wait()
        return _mock_proc(_success_stream("ok"))

    with patch("asyncio.create_subprocess_exec", side_effect=slow_proc):
        tasks = [asyncio.create_task(pool.query(f"q{i}")) for i in range(3)]

        await asyncio.sleep(0.05)

        assert len(started) == 2
        assert pool.busy_count == 2

        barrier.set()
        results = await asyncio.gather(*tasks)

    assert len(results) == 3
    for text, model, _ in results:
        assert text == "ok"


@pytest.mark.asyncio
async def test_worker_pool_timeout():
    pool = WorkerPool(size=1)

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
            with pytest.raises(asyncio.TimeoutError):
                await pool.query("hi")

    assert pool.idle_count == 1


@pytest.mark.asyncio
async def test_worker_released_on_error():
    pool = WorkerPool(size=1)

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=_mock_proc(
            _stream_bytes(
                {"type": "system", "subtype": "init", "session_id": "s"},
                {
                    "type": "result",
                    "subtype": "error",
                    "error": "boom",
                    "session_id": "s",
                },
            ),
            exit_code=1,
        ),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await pool.query("hi")

    assert pool.idle_count == 1


@pytest.mark.asyncio
async def test_model_tier_mapping():
    pool = WorkerPool(size=1)
    tier_model = {
        "fast": "claude-haiku-4-5-20251001",
        "haiku": "claude-haiku-4-5-20251001",
        "balanced": "claude-sonnet-4-6",
        "sonnet": "claude-sonnet-4-6",
        "best": "claude-opus-4-7",
        "opus": "claude-opus-4-7",
    }

    for tier, expected in tier_model.items():
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_mock_proc(_success_stream("ok")),
        ) as mock_exec:
            _, resolved, _ = await pool.query("hi", model=tier)

        assert resolved == expected, f"tier={tier}"
        assert expected in list(mock_exec.call_args[0]), f"tier={tier}"
