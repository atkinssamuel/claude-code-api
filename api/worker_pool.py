import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    DEFAULT_MODEL,
    HOOK_BASE_URL,
    MODEL_MAP,
    MODELS,
    REQUEST_TIMEOUT_S,
    TMUX_SESSION,
    WORKERS_PER_MODEL,
)

log = logging.getLogger("claude_api.worker_pool")

_PROMPT_RE = re.compile(r"^\s*[>❯╰│]")
_SEPARATOR_RE = re.compile(r"^[─\-]{10,}")
_META_RE = re.compile(r"^[✳-❀⏵]")
_BULLET_RE = re.compile(r"^⏺ ?")


# -------------------------------------------------------------------------------------
# ---------------------------------- tmux helpers -------------------------------------
# -------------------------------------------------------------------------------------


async def _tmux(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "tmux",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        log.warning(
            "_tmux: args=%s rc=%d stderr=%s",
            args,
            proc.returncode,
            stderr.decode(errors="replace")[:120],
        )

    return stdout.decode(errors="replace")


async def _capture_lines(pane_target: str) -> list[str]:
    raw = await _tmux("capture-pane", "-p", "-t", pane_target, "-S", "-")
    return raw.splitlines()


def _extract_response(all_lines: list[str], prompt: str) -> str:
    prompt_snippet = prompt[:40].strip()

    # Search backwards for the last occurrence of our prompt
    prompt_idx = -1
    for i in range(len(all_lines) - 1, -1, -1):
        if prompt_snippet in all_lines[i]:
            prompt_idx = i
            break

    if prompt_idx == -1:
        return ""

    # Collect lines after the prompt until the next prompt indicator or separator
    response_lines: list[str] = []
    for line in all_lines[prompt_idx + 1 :]:
        if _PROMPT_RE.match(line) or _SEPARATOR_RE.match(line):
            break
        if _META_RE.match(line):
            continue
        response_lines.append(_BULLET_RE.sub("", line))

    # Strip leading/trailing blanks
    while response_lines and not response_lines[0].strip():
        response_lines.pop(0)
    while response_lines and not response_lines[-1].strip():
        response_lines.pop()

    return "\n".join(response_lines).strip()


# -------------------------------------------------------------------------------------
# --------------------------------- worker / pool -------------------------------------
# -------------------------------------------------------------------------------------


@dataclass
class TmuxWorker:
    id: int
    model: str
    window_name: str
    pane_target: str
    settings_path: str
    prompt_file: str
    busy: bool = False
    startup_event: asyncio.Event = field(default_factory=asyncio.Event)
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)


class ModelPool:
    def __init__(self, model: str, workers: list[TmuxWorker]):
        self._model = model
        self._workers = workers
        self._lock = asyncio.Lock()
        self._available = asyncio.Event()

    @property
    def idle_count(self) -> int:
        return sum(1 for w in self._workers if not w.busy and w.startup_event.is_set())

    @property
    def busy_count(self) -> int:
        return sum(1 for w in self._workers if w.busy)

    @property
    def ready_count(self) -> int:
        return sum(1 for w in self._workers if w.startup_event.is_set())

    def signal_worker_ready(self, worker: TmuxWorker) -> None:
        log.info(
            "ModelPool.signal_worker_ready: model=%s worker=%d",
            self._model,
            worker.id,
        )
        worker.startup_event.set()
        self._available.set()

    async def acquire(self) -> TmuxWorker:
        log.debug(
            "ModelPool.acquire: model=%s idle=%d busy=%d",
            self._model,
            self.idle_count,
            self.busy_count,
        )

        while True:
            async with self._lock:
                for worker in self._workers:
                    if worker.startup_event.is_set() and not worker.busy:
                        worker.busy = True
                        log.debug(
                            "ModelPool.acquire: claimed worker=%d model=%s",
                            worker.id,
                            self._model,
                        )
                        return worker

                self._available.clear()

            await self._available.wait()

    def release(self, worker: TmuxWorker) -> None:
        log.debug("ModelPool.release: worker=%d", worker.id)
        worker.busy = False
        self._available.set()


# -------------------------------------------------------------------------------------
# --------------------------------- pool manager --------------------------------------
# -------------------------------------------------------------------------------------


class PoolManager:
    def __init__(self):
        self._pools: dict[str, ModelPool] = {}
        self._worker_registry: dict[int, TmuxWorker] = {}
        self._session = TMUX_SESSION

    # --------------------------------- public API ------------------------------------

    def get_worker(self, worker_id: int) -> Optional[TmuxWorker]:
        return self._worker_registry.get(worker_id)

    def get_pool(self, model: str) -> Optional[ModelPool]:
        return self._pools.get(model)

    @property
    def pools(self) -> dict[str, ModelPool]:
        return self._pools

    def signal_worker_ready(self, worker_id: int) -> None:
        worker = self._worker_registry.get(worker_id)

        if worker is None:
            log.warning("signal_worker_ready: unknown worker_id=%d", worker_id)
            return

        pool = self._pools.get(worker.model)

        if pool is None:
            log.warning("signal_worker_ready: no pool for model=%s", worker.model)
            return

        pool.signal_worker_ready(worker)

    def signal_worker_done(self, worker_id: int) -> None:
        worker = self._worker_registry.get(worker_id)

        if worker is None:
            log.warning("signal_worker_done: unknown worker_id=%d", worker_id)
            return

        log.debug("signal_worker_done: worker=%d", worker_id)
        worker.completion_event.set()

    async def query(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> tuple[str, str, int]:
        resolved_model = MODEL_MAP.get(model, MODEL_MAP[DEFAULT_MODEL])
        pool = self._pools.get(resolved_model)

        if pool is None:
            raise RuntimeError(f"no pool for model={resolved_model}")

        log.debug(
            "PoolManager.query: model=%s resolved=%s prompt=%s",
            model,
            resolved_model,
            prompt[:80],
        )

        worker = await pool.acquire()
        t0 = time.perf_counter()

        try:
            response = await asyncio.wait_for(
                self._run(worker, prompt, system),
                timeout=REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.error("PoolManager.query: timeout worker=%d", worker.id)
            raise
        except Exception as e:
            log.error(
                "PoolManager.query: error worker=%d type=%s msg=%s",
                worker.id,
                type(e).__name__,
                e,
            )
            raise
        finally:
            pool.release(worker)

        duration_ms = int((time.perf_counter() - t0) * 1000)

        log.debug(
            "PoolManager.query: done worker=%d duration_ms=%d response_len=%d",
            worker.id,
            duration_ms,
            len(response),
        )

        return response, resolved_model, duration_ms

    # --------------------------------- lifecycle -------------------------------------

    async def start(self) -> None:
        log.info(
            "PoolManager.start: session=%s workers_per_model=%d",
            self._session,
            WORKERS_PER_MODEL,
        )

        await self._reset_session()

        worker_id = 0

        for model in MODELS:
            short = model.split("-")[1]
            workers: list[TmuxWorker] = []

            for i in range(WORKERS_PER_MODEL):
                window_name = f"{short}-{i}"
                pane_target = f"{self._session}:{window_name}"

                worker = TmuxWorker(
                    id=worker_id,
                    model=model,
                    window_name=window_name,
                    pane_target=pane_target,
                    settings_path=f"/tmp/claude-worker-{worker_id}-settings.json",
                    prompt_file=f"/tmp/claude-worker-{worker_id}-prompt.txt",
                )

                self._worker_registry[worker_id] = worker
                workers.append(worker)

                await self._launch_worker(worker, is_first=(worker_id == 0))

                worker_id += 1

            self._pools[model] = ModelPool(model=model, workers=workers)

        log.info(
            "PoolManager.start: launched %d workers across %d models, awaiting SessionStart hooks",
            len(self._worker_registry),
            len(MODELS),
        )

    async def stop(self) -> None:
        log.info("PoolManager.stop: killing session=%s", self._session)
        await _tmux("kill-session", "-t", self._session)

    # --------------------------------- internals -------------------------------------

    async def _reset_session(self) -> None:
        await _tmux("kill-session", "-t", self._session)
        await _tmux(
            "new-session",
            "-d",
            "-s",
            self._session,
            "-x",
            "220",
            "-y",
            "50",
        )
        await _tmux("set-option", "-t", self._session, "history-limit", "100000")
        log.debug("_reset_session: session=%s created", self._session)

    async def _launch_worker(self, worker: TmuxWorker, is_first: bool) -> None:
        self._write_settings(worker)

        if is_first:
            await _tmux("rename-window", "-t", f"{self._session}:0", worker.window_name)
        else:
            await _tmux("new-window", "-t", self._session, "-n", worker.window_name)

        claude_cmd = (
            f"claude "
            f"--model {worker.model} "
            f"--settings {worker.settings_path} "
            f"--dangerously-skip-permissions"
        )

        await _tmux("send-keys", "-t", worker.pane_target, claude_cmd, "Enter")

        log.debug(
            "_launch_worker: worker=%d pane=%s model=%s",
            worker.id,
            worker.pane_target,
            worker.model,
        )

    def _write_settings(self, worker: TmuxWorker) -> None:
        hook_base = f"{HOOK_BASE_URL}/internal/hook?worker_id={worker.id}"

        settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f"curl -sf -X POST "
                                    f"'{hook_base}&event=start' "
                                    f"-H 'Content-Type: application/json' "
                                    f"-d @-"
                                ),
                            }
                        ]
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f"curl -sf -X POST "
                                    f"'{hook_base}&event=stop' "
                                    f"-H 'Content-Type: application/json' "
                                    f"-d @-"
                                ),
                            }
                        ]
                    }
                ],
            }
        }

        with open(worker.settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        log.debug("_write_settings: worker=%d path=%s", worker.id, worker.settings_path)

    async def _run(self, worker: TmuxWorker, prompt: str, system: Optional[str]) -> str:
        worker.completion_event.clear()

        effective_prompt = (
            f"[System instructions: {system}]\n\n{prompt}" if system else prompt
        )

        log.debug(
            "_run: worker=%d prompt=%s",
            worker.id,
            effective_prompt[:80],
        )

        await self._send_prompt(worker, effective_prompt)

        log.debug("_run: worker=%d prompt sent, awaiting Stop hook", worker.id)

        await worker.completion_event.wait()

        all_lines = await _capture_lines(worker.pane_target)
        response = _extract_response(all_lines, effective_prompt)

        log.debug(
            "_run: worker=%d all_lines=%d response_len=%d",
            worker.id,
            len(all_lines),
            len(response),
        )

        return response

    async def _send_prompt(self, worker: TmuxWorker, prompt: str) -> None:
        with open(worker.prompt_file, "w") as f:
            f.write(prompt + "\n")

        buf_name = f"w{worker.id}"
        await _tmux("load-buffer", "-b", buf_name, worker.prompt_file)
        await _tmux("paste-buffer", "-b", buf_name, "-t", worker.pane_target)

        log.debug("_send_prompt: worker=%d buf=%s", worker.id, buf_name)
