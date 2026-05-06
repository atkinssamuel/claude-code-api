import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from api.config import MODELS
from api.worker_pool import ModelPool, PoolManager, TmuxWorker, _read_transcript


# -------------------------------------------------------------------------------------
# ---------------------------------- helpers ------------------------------------------
# -------------------------------------------------------------------------------------


def _make_manager(workers_per_model: int = 1) -> PoolManager:
    """PoolManager with pre-built, pre-ready workers — no tmux calls needed."""
    manager = PoolManager()
    worker_id = 0

    for model in MODELS:
        short = model.split("-")[1]
        workers: list[TmuxWorker] = []

        for i in range(workers_per_model):
            worker = TmuxWorker(
                id=worker_id,
                model=model,
                window_name=f"{short}-{i}",
                pane_target=f"test:{short}-{i}",
                settings_path=f"/tmp/test-worker-{worker_id}.json",
                prompt_file=f"/tmp/test-prompt-{worker_id}.txt",
            )
            worker.startup_event.set()
            workers.append(worker)
            manager._worker_registry[worker_id] = worker
            worker_id += 1

        manager._pools[model] = ModelPool(model=model, workers=workers)

    return manager


def _instant_run(response: str = "ok"):
    """Returns an async _run replacement that completes immediately."""

    async def _inner(worker: TmuxWorker, prompt: str, system) -> str:
        return response

    return _inner


# -------------------------------------------------------------------------------------
# ----------------------------------- unit tests --------------------------------------
# -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_and_ready_counts():
    manager = _make_manager(workers_per_model=2)

    for model in MODELS:
        pool = manager.get_pool(model)
        assert pool.idle_count == 2
        assert pool.busy_count == 0
        assert pool.ready_count == 2


@pytest.mark.asyncio
async def test_query_routes_to_correct_model():
    manager = _make_manager()

    with patch.object(manager, "_run", _instant_run("4")):
        _, resolved_model, _ = await manager.query("What is 2+2?", model="fast")

    assert resolved_model == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_model_tier_aliases():
    tier_to_model = {
        "fast": "claude-haiku-4-5-20251001",
        "haiku": "claude-haiku-4-5-20251001",
        "balanced": "claude-sonnet-4-6",
        "sonnet": "claude-sonnet-4-6",
        "best": "claude-opus-4-7",
        "opus": "claude-opus-4-7",
    }

    manager = _make_manager()

    for tier, expected_model in tier_to_model.items():
        with patch.object(manager, "_run", _instant_run("ok")):
            _, resolved, _ = await manager.query("hi", model=tier)

        assert resolved == expected_model, f"tier={tier}"


@pytest.mark.asyncio
async def test_worker_released_after_query():
    manager = _make_manager()
    pool = manager.get_pool("claude-haiku-4-5-20251001")

    with patch.object(manager, "_run", _instant_run("answer")):
        await manager.query("hi", model="fast")

    assert pool.idle_count == 1
    assert pool.busy_count == 0


@pytest.mark.asyncio
async def test_worker_released_on_error():
    manager = _make_manager()
    pool = manager.get_pool("claude-haiku-4-5-20251001")

    async def failing_run(worker, prompt, system):
        raise RuntimeError("boom")

    with patch.object(manager, "_run", failing_run):
        with pytest.raises(RuntimeError):
            await manager.query("hi", model="fast")

    assert pool.idle_count == 1


@pytest.mark.asyncio
async def test_pool_queues_when_exhausted():
    """Extra requests wait and succeed once a slot is freed."""
    manager = _make_manager(workers_per_model=1)
    pool = manager.get_pool("claude-haiku-4-5-20251001")
    gate = asyncio.Event()

    async def gated_run(worker, prompt, system):
        await gate.wait()
        return "ok"

    with patch.object(manager, "_run", gated_run):
        tasks = [
            asyncio.create_task(manager.query(f"q{i}", model="fast")) for i in range(3)
        ]

        await asyncio.sleep(0)
        assert pool.busy_count == 1

        gate.set()
        results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert all(r[0] == "ok" for r in results)
    assert pool.idle_count == 1


@pytest.mark.asyncio
async def test_timeout_releases_worker():
    manager = _make_manager()
    pool = manager.get_pool("claude-haiku-4-5-20251001")

    async def slow_run(worker, prompt, system):
        await asyncio.sleep(999)
        return "ok"

    with (
        patch.object(manager, "_run", slow_run),
        patch("api.worker_pool.REQUEST_TIMEOUT_S", 0.05),
    ):
        with pytest.raises(asyncio.TimeoutError):
            await manager.query("hi", model="fast")

    assert pool.idle_count == 1


@pytest.mark.asyncio
async def test_signal_worker_ready():
    manager = _make_manager(workers_per_model=1)

    for w in manager._worker_registry.values():
        w.startup_event.clear()

    haiku_pool = manager.get_pool("claude-haiku-4-5-20251001")
    assert haiku_pool.ready_count == 0

    manager.signal_worker_ready(0)

    assert haiku_pool.ready_count == 1
    assert haiku_pool.idle_count == 1


@pytest.mark.asyncio
async def test_signal_worker_done():
    manager = _make_manager()
    worker = manager._worker_registry[0]
    worker.completion_event.clear()

    manager.signal_worker_done(0, "/tmp/fake-transcript.jsonl")

    assert worker.completion_event.is_set()
    assert worker.last_transcript_path == "/tmp/fake-transcript.jsonl"


@pytest.mark.asyncio
async def test_run_uses_send_prompt_then_awaits_hook():
    """_run: sends prompt, waits for completion_event, reads response via _read_response."""
    manager = _make_manager()
    worker = manager._worker_registry[0]

    send_called = False

    async def patched_send(w, prompt):
        nonlocal send_called
        send_called = True
        w.last_transcript_path = "/tmp/fake.jsonl"
        w.completion_event.set()

    with (
        patch.object(manager, "_send_prompt", patched_send),
        patch.object(manager, "_read_response", AsyncMock(return_value="the answer")),
    ):
        response = await manager._run(worker, "the prompt", None)

    assert send_called
    assert response == "the answer"


# -------------------------------------------------------------------------------------
# ----------------------------- transcript parsing tests ------------------------------
# -------------------------------------------------------------------------------------


def _assistant_entry(text: str) -> str:
    import json

    return json.dumps(
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}
    )


def test_read_transcript_basic(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type": "user", "message": {"role": "user", "content": "hi"}}\n'
        + _assistant_entry("hello")
        + "\n"
    )
    assert _read_transcript(str(transcript)) == "hello"


def test_read_transcript_content_block(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_assistant_entry("pong") + "\n")
    assert _read_transcript(str(transcript)) == "pong"


def test_read_transcript_skips_thinking_blocks(tmp_path):
    import json

    transcript = tmp_path / "t.jsonl"
    entry = json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "...", "signature": "x"},
                    {"type": "text", "text": "pong"},
                ],
            },
        }
    )
    transcript.write_text(entry + "\n")
    assert _read_transcript(str(transcript)) == "pong"


def test_read_transcript_returns_last_assistant(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _assistant_entry("first")
        + "\n"
        + '{"type": "user", "message": {"role": "user", "content": "follow-up"}}\n'
        + _assistant_entry("second")
        + "\n"
    )
    assert _read_transcript(str(transcript)) == "second"


def test_read_transcript_missing_file():
    result = _read_transcript("/tmp/does-not-exist-xyz.jsonl")
    assert result == ""
