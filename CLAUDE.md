# claude-code-api

A forever-running local HTTP API that proxies requests to the `claude` CLI binary,
letting you use your Claude Code subscription for local dev/testing instead of paying
Anthropic SDK token fees.

## What it does

Exposes a simple REST endpoint. Callers send a prompt plus model-tier parameters;
the server routes the request to a pre-warmed `claude` worker process and streams or
returns the response.

## Architecture

```
Client (curl / test script)
        |
        v
  FastAPI server  (port 8731)
        |
        v
  Worker Pool  — N pre-spawned `claude` CLI processes kept alive between requests
        |
        v
  claude binary  (authenticated via ~/.claude credentials mounted into the container)
```

**Worker pool:** A fixed-size pool of `claude` subprocess sessions. Each session stays
alive between requests to avoid cold-start latency. Requests queue against the pool;
idle sessions are claimed immediately.

**Model tiers:** Map a caller-supplied `model` parameter to the appropriate
`--model` flag passed to the `claude` CLI:

| Tier | CLI flag |
|------|----------|
| `fast` / `haiku` | `claude-haiku-4-5-20251001` |
| `balanced` / `sonnet` | `claude-sonnet-4-6` |
| `best` / `opus` | `claude-opus-4-7` |

## Tech stack

- **Python 3.12 + FastAPI** — HTTP server
- **asyncio subprocess** — non-blocking worker pool
- **Docker + Docker Compose** — containerized, `restart: always` so it survives reboots
- **pytest + httpx** — test suite

## Running locally

```bash
docker compose up --build -d
curl -X POST http://localhost:8731/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello", "model": "fast"}'
```

## API

### `POST /query`

```json
{
  "prompt": "Your prompt here",
  "model": "fast | balanced | best",   // optional, default: balanced
  "max_tokens": 1024,                  // optional
  "system": "You are a helpful assistant."  // optional
}
```

Response:

```json
{
  "response": "...",
  "model": "claude-haiku-4-5-20251001",
  "duration_ms": 312
}
```

### `GET /health`

Returns `{"status": "ok", "workers": {"idle": 3, "busy": 1}}`.

## Project structure

```
claude-code-api/
  api/
    main.py          # FastAPI app + lifespan (pool startup/shutdown)
    worker_pool.py   # async subprocess worker pool
    models.py        # request/response schemas
    config.py        # env vars (pool size, timeouts, model map)
  tests/
    test_api.py
    test_worker_pool.py
  Dockerfile
  docker-compose.yml
  .env.example
```

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `POOL_SIZE` | `4` | Number of pre-warmed `claude` workers |
| `REQUEST_TIMEOUT_S` | `60` | Per-request timeout in seconds |
| `DEFAULT_MODEL` | `balanced` | Default model tier |

## Tests

```bash
# inside container or with deps installed
pytest tests/ -v
```

Tests cover: health endpoint, successful query round-trip, model tier routing,
pool exhaustion behaviour, and request timeout handling.

## Worker communication via hooks

Worker sessions **must** use the `claude` CLI's built-in hook system to signal completion and
deliver response content — do **not** parse raw stdout or use ad-hoc delimiters.

### How it works

Each worker is launched with `--output-format stream-json --include-hook-events`. This produces
a newline-delimited JSON stream containing both the response tokens and hook lifecycle events.
The two key event types are:

- **`content_block_delta`** — carries incremental response text
- **`message_stop`** / **`Stop` hook event** — signals the response is fully complete

Alternatively (or in addition), configure HTTP hooks so that `claude` POSTs back to a local
FastAPI endpoint when a session stops:

```json
{
  "hooks": {
    "Stop": [{
      "hooks": [{
        "type": "http",
        "url": "http://localhost:8731/internal/hook?worker_id=<N>"
      }]
    }]
  }
}
```

Each worker gets its own settings file (passed via `--settings /tmp/worker-<N>-settings.json`)
so the `worker_id` query param routes the callback back to the correct in-flight request.

### Hook event payload (Stop)

```json
{
  "session_id": "abc123",
  "hook_event_name": "Stop",
  "cwd": "/workspace"
}
```

The full response text is captured from the `content_block_delta` events in the stream; the
`Stop` hook event (or `message_stop` stream event) is the authoritative completion signal.

### Why hooks, not stdout parsing

- Hooks fire exactly once per completed response — no ambiguity about where a response ends.
- HTTP hooks decouple the worker process from the pool manager; the pool never needs to poll.
- `--include-hook-events` in the stream lets a single reader handle both content and lifecycle
  signals without a separate out-of-band channel.

## Key constraints

- Never use the Anthropic Python/JS SDK — all LLM calls go through the `claude` CLI binary.
- Never use `sleep`/`time.sleep` for sequencing — use `asyncio.wait_for`, queues, and events.
- The `~/.claude` directory (credentials) must be bind-mounted into the container at runtime.
- Keep the worker pool hot; do not tear down sessions between requests.
- All completion detection and response collection must go through the hook/stream-json mechanism
  described above — never rely on EOF or silent stdout to determine when a query is done.
