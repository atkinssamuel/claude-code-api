import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from api.config import MODELS
from api.worker_pool import ModelPool, PoolManager, TmuxWorker, _extract_response


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

    manager.signal_worker_done(0)

    assert worker.completion_event.is_set()


@pytest.mark.asyncio
async def test_run_uses_send_prompt_then_awaits_hook():
    """_run: sends prompt, waits on completion_event, then captures lines."""
    manager = _make_manager()
    worker = manager._worker_registry[0]

    send_called = False

    async def patched_send(w, prompt):
        nonlocal send_called
        send_called = True

        # Fire the Stop hook synchronously so _run can proceed
        w.completion_event.set()

    lines_calls: list[str] = []

    async def patched_capture(pane_target):
        lines_calls.append(pane_target)
        return ["old line", "the prompt", "the answer", "❯ "]

    with (
        patch.object(manager, "_send_prompt", patched_send),
        patch("api.worker_pool._capture_lines", patched_capture),
    ):
        response = await manager._run(worker, "the prompt", None)

    assert send_called
    assert len(lines_calls) == 1
    assert response == "the answer"


# -------------------------------------------------------------------------------------
# ---------------------------- response extraction tests ------------------------------
# -------------------------------------------------------------------------------------


def test_extract_response_basic():
    all_lines = [
        "old line 1",
        "old line 2",
        "What is 2+2?",
        "2 + 2 = 4",
        "❯ ",
    ]
    result = _extract_response(all_lines, "What is 2+2?")
    assert result == "2 + 2 = 4"


def test_extract_response_strips_leading_blanks():
    all_lines = [
        "old",
        "prompt text here",
        "",
        "the answer",
        "❯ ",
    ]
    result = _extract_response(all_lines, "prompt text here")
    assert result == "the answer"


def test_extract_response_multiline():
    all_lines = [
        "old",
        "my prompt",
        "line one",
        "line two",
        "line three",
        "❯ ",
    ]
    result = _extract_response(all_lines, "my prompt")
    assert "line one" in result
    assert "line three" in result


def test_extract_response_no_prompt_echo():
    """Returns empty when prompt not found in pane."""
    all_lines = [
        "something unrelated",
        "actual response",
        "❯ ",
    ]
    result = _extract_response(all_lines, "completely different prompt xyz123")
    assert result == ""


def test_extract_response_prompt_indicator_variants():
    for indicator in [">", "> ", "╰─>", "│", "❯", "❯ "]:
        all_lines = ["old", "prompt", "response text", indicator]
        result = _extract_response(all_lines, "prompt")
        assert "response text" in result, f"indicator={indicator!r}"
        assert indicator not in result, f"indicator={indicator!r} not stripped"


def test_extract_response_strips_bullet_prefix():
    all_lines = [
        "my prompt",
        "⏺ pong",
        "❯ ",
    ]
    result = _extract_response(all_lines, "my prompt")
    assert result == "pong"


def test_extract_response_skips_meta_lines():
    all_lines = [
        "my prompt",
        "⏺ the answer",
        "✻ Sautéed for 1s",
        "❯ ",
    ]
    result = _extract_response(all_lines, "my prompt")
    assert result == "the answer"
    assert "Sautéed" not in result


def test_extract_response_separator_stops_collection():
    all_lines = [
        "my prompt",
        "⏺ the answer",
        "──────────────────────────────",
        "❯ ",
    ]
    result = _extract_response(all_lines, "my prompt")
    assert result == "the answer"


def test_extract_response_uses_last_occurrence():
    """When the same prompt appears twice, extract after the last one."""
    all_lines = [
        "my prompt",
        "first answer",
        "❯ ",
        "my prompt",
        "second answer",
        "❯ ",
    ]
    result = _extract_response(all_lines, "my prompt")
    assert result == "second answer"
