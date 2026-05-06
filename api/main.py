import asyncio
import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from .config import DEFAULT_MODEL, MODEL_MAP, MODELS
from .models import HealthResponse, ModelWorkerStatus, QueryRequest, QueryResponse
from .worker_pool import PoolManager

# -------------------------------------------------------------------------------------
# ----------------------------------- logging -----------------------------------------
# -------------------------------------------------------------------------------------

_default_log_dir = (
    "/var/log/claude-api"
    if os.access("/var/log", os.W_OK)
    else os.path.expanduser("~/Library/Logs/claude-api")
)
_log_dir = os.environ.get("LOG_DIR", _default_log_dir)
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, "api.log")

print(f"Logging to: {_log_path}", flush=True)

_fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_file_handler = logging.handlers.RotatingFileHandler(
    _log_path, maxBytes=5 * 1024 * 1024, backupCount=3
)
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _console_handler])

log = logging.getLogger("claude_api.main")

# -------------------------------------------------------------------------------------
# ------------------------------------- app -------------------------------------------
# -------------------------------------------------------------------------------------

_pool_manager: Optional[PoolManager] = None


def get_pool_manager() -> PoolManager:
    if _pool_manager is None:
        raise HTTPException(status_code=503, detail="pool not ready")
    return _pool_manager


def get_optional_pool_manager() -> Optional[PoolManager]:
    return _pool_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_manager

    log.info("lifespan: creating pool manager")
    _pool_manager = PoolManager()

    # Launch workers in background so the HTTP server is already listening
    # when SessionStart hooks POST back in.
    asyncio.create_task(_pool_manager.start())

    log.info("lifespan: pool manager started, workers launching in background")

    yield

    log.info("lifespan: shutting down")

    if _pool_manager is not None:
        await _pool_manager.stop()

    _pool_manager = None


app = FastAPI(title="claude-code-api", version="2.0.0", lifespan=lifespan)

# -------------------------------------------------------------------------------------
# ----------------------------------- routes ------------------------------------------
# -------------------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(manager: Optional[PoolManager] = Depends(get_optional_pool_manager)):
    log.debug("health: called")

    if manager is None:
        return HealthResponse(status="starting", workers={})

    workers: dict[str, ModelWorkerStatus] = {}

    for model, pool in manager.pools.items():
        workers[model] = ModelWorkerStatus(
            idle=pool.idle_count,
            busy=pool.busy_count,
            ready=pool.ready_count,
        )

    result = HealthResponse(status="ok", workers=workers)

    log.debug("health: pools=%d", len(workers))

    return result


@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    manager: PoolManager = Depends(get_pool_manager),
):
    log.debug(
        "query: prompt=%s model=%s system=%s",
        req.prompt[:80],
        req.model,
        req.system[:80] if req.system else None,
    )

    try:
        response, resolved_model, duration_ms = await manager.query(
            prompt=req.prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            system=req.system,
        )
    except asyncio.TimeoutError:
        log.error("query: timeout prompt=%s", req.prompt[:80])
        raise HTTPException(status_code=504, detail="request timed out")
    except RuntimeError as e:
        log.error("query: runtime_error msg=%s", e)
        raise HTTPException(status_code=500, detail=str(e))

    result = QueryResponse(
        response=response,
        model=resolved_model,
        duration_ms=duration_ms,
    )

    log.debug(
        "query: done model=%s duration_ms=%d response_len=%d",
        resolved_model,
        duration_ms,
        len(response),
    )

    return result


@app.post("/internal/hook")
async def internal_hook(
    worker_id: int = Query(...),
    event: str = Query(...),
    request: Request = None,
    manager: Optional[PoolManager] = Depends(get_optional_pool_manager),
):
    log.debug("internal_hook: worker_id=%d event=%s", worker_id, event)

    if manager is None:
        log.warning(
            "internal_hook: manager not ready, ignoring worker_id=%d", worker_id
        )
        return {}

    if event == "start":
        manager.signal_worker_ready(worker_id)
    elif event == "stop":
        manager.signal_worker_done(worker_id)
    else:
        log.warning("internal_hook: unknown event=%s worker_id=%d", event, worker_id)

    return {}
