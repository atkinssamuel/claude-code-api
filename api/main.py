import asyncio
import logging
import logging.handlers
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from .config import POOL_SIZE
from .models import HealthResponse, QueryRequest, QueryResponse, WorkerStatus
from .worker_pool import WorkerPool

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

_pool: WorkerPool | None = None


def get_pool() -> WorkerPool:
    if _pool is None:
        raise HTTPException(status_code=503, detail="pool not initialized")
    return _pool


def get_optional_pool() -> WorkerPool | None:
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool

    log.info("lifespan: starting pool size=%d", POOL_SIZE)
    _pool = WorkerPool(size=POOL_SIZE)
    log.info("lifespan: pool ready idle=%d", _pool.idle_count)

    yield

    log.info("lifespan: shutting down")
    _pool = None


app = FastAPI(title="claude-code-api", version="1.0.0", lifespan=lifespan)

# -------------------------------------------------------------------------------------
# ----------------------------------- routes ------------------------------------------
# -------------------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(pool: WorkerPool | None = Depends(get_optional_pool)):
    log.debug("health: called")

    if pool is None:
        log.warning("health: pool not initialized")
        return HealthResponse(status="starting", workers=WorkerStatus(idle=0, busy=0))

    result = HealthResponse(
        status="ok",
        workers=WorkerStatus(idle=pool.idle_count, busy=pool.busy_count),
    )

    log.debug("health: idle=%d busy=%d", result.workers.idle, result.workers.busy)

    return result


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, pool: WorkerPool = Depends(get_pool)):
    log.debug(
        "query: prompt=%s model=%s max_tokens=%s system=%s",
        req.prompt[:80],
        req.model,
        req.max_tokens,
        req.system[:80] if req.system else None,
    )

    try:
        response, resolved_model, duration_ms = await pool.query(
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
