import asyncio
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.config import MODELS
from api.worker_pool import ModelPool, PoolManager, TmuxWorker


# -------------------------------------------------------------------------------------
# ---------------------------------- helpers ------------------------------------------
# -------------------------------------------------------------------------------------


def _make_manager(workers_per_model: int = 1) -> PoolManager:
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
    async def _inner(worker: TmuxWorker, prompt: str, system) -> str:
        return response

    return _inner


# -------------------------------------------------------------------------------------
# ---------------------------------- fixtures -----------------------------------------
# -------------------------------------------------------------------------------------


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture()
async def client():
    from api.main import app, get_optional_pool_manager, get_pool_manager

    manager = _make_manager(workers_per_model=1)
    app.dependency_overrides[get_pool_manager] = lambda: manager
    app.dependency_overrides[get_optional_pool_manager] = lambda: manager

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
    assert set(body["workers"].keys()) == set(MODELS)

    for model, counts in body["workers"].items():
        assert counts["idle"] + counts["busy"] == 1
        assert counts["ready"] == 1


@pytest.mark.anyio
async def test_health_starting():
    from api.main import app, get_optional_pool_manager

    app.dependency_overrides[get_optional_pool_manager] = lambda: None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.get("/health")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["status"] == "starting"


@pytest.mark.anyio
async def test_query_success(client):
    from api.main import app, get_pool_manager

    manager = _make_manager()
    app.dependency_overrides[get_pool_manager] = lambda: manager

    with patch.object(manager, "_run", _instant_run("4")):
        resp = await client.post(
            "/query", json={"prompt": "what is 2+2", "model": "fast"}
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "4"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["duration_ms"] >= 0


@pytest.mark.anyio
async def test_query_model_routing(client):
    from api.main import app, get_pool_manager

    tier_to_model = {
        "fast": "claude-haiku-4-5-20251001",
        "haiku": "claude-haiku-4-5-20251001",
        "balanced": "claude-sonnet-4-6",
        "sonnet": "claude-sonnet-4-6",
        "best": "claude-opus-4-7",
        "opus": "claude-opus-4-7",
    }

    for tier, expected_model in tier_to_model.items():
        manager = _make_manager()
        app.dependency_overrides[get_pool_manager] = lambda m=manager: m

        with patch.object(manager, "_run", _instant_run("ok")):
            resp = await client.post("/query", json={"prompt": "hi", "model": tier})

        assert resp.status_code == 200, f"tier={tier}"
        assert resp.json()["model"] == expected_model, f"tier={tier}"

    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_query_default_model(client):
    from api.main import app, get_pool_manager

    manager = _make_manager()
    app.dependency_overrides[get_pool_manager] = lambda: manager

    with patch.object(manager, "_run", _instant_run("ok")):
        resp = await client.post("/query", json={"prompt": "hi"})

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-4-6"


@pytest.mark.anyio
async def test_query_timeout(client):
    from api.main import app, get_pool_manager

    manager = _make_manager()
    app.dependency_overrides[get_pool_manager] = lambda: manager

    async def slow_run(worker, prompt, system):
        await asyncio.sleep(999)
        return "ok"

    with (
        patch.object(manager, "_run", slow_run),
        patch("api.worker_pool.REQUEST_TIMEOUT_S", 0.05),
    ):
        resp = await client.post("/query", json={"prompt": "hi"})

    app.dependency_overrides.clear()
    assert resp.status_code == 504


@pytest.mark.anyio
async def test_query_503_when_pool_not_ready():
    from api.main import app, get_pool_manager

    app.dependency_overrides[get_pool_manager] = lambda: (_ for _ in ()).throw(
        __import__("fastapi").HTTPException(status_code=503, detail="pool not ready")
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.post("/query", json={"prompt": "hi"})

    app.dependency_overrides.clear()
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_internal_hook_start(client):
    from api.main import app, get_optional_pool_manager

    manager = _make_manager()
    manager._worker_registry[0].startup_event.clear()
    app.dependency_overrides[get_optional_pool_manager] = lambda: manager

    resp = await client.post("/internal/hook?worker_id=0&event=start")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert manager._worker_registry[0].startup_event.is_set()


@pytest.mark.anyio
async def test_internal_hook_stop(client):
    from api.main import app, get_optional_pool_manager

    manager = _make_manager()
    manager._worker_registry[0].completion_event.clear()
    app.dependency_overrides[get_optional_pool_manager] = lambda: manager

    with patch("api.worker_pool._read_transcript", return_value="ok"):
        resp = await client.post(
            "/internal/hook?worker_id=0&event=stop",
            json={"transcript_path": "/tmp/fake.jsonl"},
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert manager._worker_registry[0].completion_event.is_set()


@pytest.mark.anyio
async def test_pool_exhaustion_queues(client):
    from api.main import app, get_pool_manager

    # Pool-level exhaustion queuing is unit-tested in test_worker_pool.py.
    # Here we verify that all concurrent HTTP requests eventually succeed.
    manager = _make_manager(workers_per_model=2)
    app.dependency_overrides[get_pool_manager] = lambda: manager

    with patch.object(manager, "_run", _instant_run("ok")):
        responses = await asyncio.gather(
            *[client.post("/query", json={"prompt": f"q{i}"}) for i in range(6)]
        )

    app.dependency_overrides.clear()
    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["response"] == "ok" for r in responses)
