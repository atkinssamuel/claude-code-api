import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .config import DEFAULT_MODEL, MODEL_MAP, POOL_SIZE, REQUEST_TIMEOUT_S

log = logging.getLogger("claude_api.worker_pool")


@dataclass
class Worker:
    id: int
    session_id: Optional[str] = None
    busy: bool = False


class WorkerPool:
    def __init__(self, size: int = POOL_SIZE):
        self._workers: list[Worker] = [Worker(id=i) for i in range(size)]
        self._lock = asyncio.Lock()
        self._available = asyncio.Event()
        self._available.set()
        log.info("WorkerPool.__init__: size=%d", size)

    @property
    def idle_count(self) -> int:
        return sum(1 for w in self._workers if not w.busy)

    @property
    def busy_count(self) -> int:
        return sum(1 for w in self._workers if w.busy)

    async def query(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> tuple[str, str, int]:
        log.debug(
            "WorkerPool.query: prompt=%s model=%s max_tokens=%s system=%s",
            prompt[:80],
            model,
            max_tokens,
            system[:80] if system else None,
        )

        worker = await self._acquire()

        t0 = time.perf_counter()

        try:
            response, resolved_model = await asyncio.wait_for(
                self._run(worker, prompt, model, max_tokens, system),
                timeout=REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.error(
                "WorkerPool.query: timeout worker=%d after %ds",
                worker.id,
                REQUEST_TIMEOUT_S,
            )
            raise
        except Exception as e:
            log.error(
                "WorkerPool.query: error worker=%d type=%s msg=%s",
                worker.id,
                type(e).__name__,
                e,
            )
            raise
        finally:
            self._release(worker)

        duration_ms = int((time.perf_counter() - t0) * 1000)

        log.debug(
            "WorkerPool.query: done worker=%d duration_ms=%d response_len=%d",
            worker.id,
            duration_ms,
            len(response),
        )

        return response, resolved_model, duration_ms

    async def _acquire(self) -> Worker:
        log.debug("WorkerPool._acquire: waiting idle=%d", self.idle_count)

        while True:
            async with self._lock:
                for worker in self._workers:
                    if not worker.busy:
                        worker.busy = True
                        log.debug("WorkerPool._acquire: claimed worker=%d", worker.id)
                        return worker

                self._available.clear()

            await self._available.wait()

    def _release(self, worker: Worker) -> None:
        log.debug("WorkerPool._release: worker=%d", worker.id)
        worker.busy = False
        self._available.set()

    async def _run(
        self,
        worker: Worker,
        prompt: str,
        model_tier: str,
        max_tokens: Optional[int],
        system: Optional[str],
    ) -> tuple[str, str]:
        resolved_model = MODEL_MAP.get(model_tier, MODEL_MAP[DEFAULT_MODEL])

        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--include-hook-events",
            "--verbose",
            "--model",
            resolved_model,
            "--dangerously-skip-permissions",
        ]

        if worker.session_id and not system:
            cmd += ["--resume", worker.session_id]

        if system:
            cmd += ["--system-prompt", system]

        log.debug(
            "WorkerPool._run: worker=%d model=%s session_id=%s",
            worker.id,
            resolved_model,
            worker.session_id,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        response_text = ""
        session_id = worker.session_id
        error_text = ""

        assert proc.stdout is not None

        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log.warning(
                    "WorkerPool._run: worker=%d json_decode_error line=%s",
                    worker.id,
                    line[:120],
                )
                continue

            event_type = event.get("type")

            if event_type == "system":
                session_id = event.get("session_id", session_id)
                log.debug(
                    "WorkerPool._run: worker=%d system_init session_id=%s",
                    worker.id,
                    session_id,
                )

            elif event_type == "assistant":
                message = event.get("message", {})
                for block in message.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        response_text += block.get("text", "")

                log.debug(
                    "WorkerPool._run: worker=%d assistant accumulated_len=%d",
                    worker.id,
                    len(response_text),
                )

            elif event_type == "result":
                session_id = event.get("session_id", session_id)
                subtype = event.get("subtype")

                if subtype == "error":
                    error_text = event.get(
                        "error", str(event.get("result", "unknown error"))
                    )
                    log.error(
                        "WorkerPool._run: worker=%d result_error=%s",
                        worker.id,
                        error_text,
                    )
                elif subtype == "success" and not response_text:
                    response_text = event.get("result", "")

                log.debug(
                    "WorkerPool._run: worker=%d result subtype=%s session_id=%s",
                    worker.id,
                    subtype,
                    session_id,
                )

            else:
                log.debug(
                    "WorkerPool._run: worker=%d ignored event_type=%s",
                    worker.id,
                    event_type,
                )

        assert proc.stderr is not None
        stderr_bytes = await proc.stderr.read()
        stderr_output = stderr_bytes.decode(errors="replace").strip()
        exit_code = await proc.wait()

        log.debug(
            "WorkerPool._run: worker=%d exit_code=%d stderr_len=%d",
            worker.id,
            exit_code,
            len(stderr_output),
        )

        if stderr_output:
            log.warning(
                "WorkerPool._run: worker=%d stderr=%s",
                worker.id,
                stderr_output[:200],
            )

        if error_text:
            raise RuntimeError(f"claude error: {error_text}")

        if exit_code != 0 and not response_text:
            raise RuntimeError(
                f"claude exited with code {exit_code}: {stderr_output[:200]}"
            )

        worker.session_id = session_id

        log.debug(
            "WorkerPool._run: worker=%d completed session_id=%s response_len=%d",
            worker.id,
            session_id,
            len(response_text),
        )

        return response_text, resolved_model
