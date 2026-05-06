"""
Timing analysis for the claude-code-api across model tiers, context sizes, and prompt types.

Usage:
    python timer/timer.py                    # run all scenarios
    python timer/timer.py --tags long_ctx    # filter by tag
    python timer/timer.py --model fast       # filter by model tier
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import SCENARIOS

# -------------------------------------------------------------------------------------
# ------------------------------------ logging ----------------------------------------
# -------------------------------------------------------------------------------------

_log_dir = os.path.expanduser("~/Library/Logs/claude-code-api-timer")
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, "timer.log")

print(f"Logging to: {_log_path}")

_handler = RotatingFileHandler(_log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
))

log = logging.getLogger("timer")
log.addHandler(_handler)
log.setLevel(logging.DEBUG)

# -------------------------------------------------------------------------------------
# --------------------------------- configuration -------------------------------------
# -------------------------------------------------------------------------------------

BASE_URL = "http://localhost:8731"
QUERY_URL = f"{BASE_URL}/query"
HEALTH_URL = f"{BASE_URL}/health"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------------------
# ----------------------------------- api calls --------------------------------------
# -------------------------------------------------------------------------------------

def check_health() -> dict:
    log.debug("check_health: sending GET %s", HEALTH_URL)
    t0 = time.perf_counter()

    resp = requests.get(HEALTH_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    log.debug("check_health: completed in %.1fms status=%s", (time.perf_counter() - t0) * 1000, data)
    return data


def run_query(scenario: dict) -> dict:
    name = scenario["name"]
    payload = {
        "prompt": scenario["prompt"],
        "model": scenario["model"],
        "max_tokens": scenario["max_tokens"],
    }

    if scenario.get("system"):
        payload["system"] = scenario["system"]

    prompt_preview = scenario["prompt"][:80].replace("\n", " ")
    log.debug(
        "run_query: name=%s model=%s max_tokens=%d prompt_preview=%s…",
        name, scenario["model"], scenario["max_tokens"], prompt_preview,
    )

    t0 = time.perf_counter()
    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        wall_ms = (time.perf_counter() - t0) * 1000

        result = {
            "scenario": name,
            "model_tier": scenario["model"],
            "tags": scenario["tags"],
            "prompt_chars": len(scenario["prompt"]),
            "system_chars": len(scenario.get("system") or ""),
            "max_tokens": scenario["max_tokens"],
            "model_reported": data.get("model"),
            "duration_ms_reported": data.get("duration_ms"),
            "duration_ms_wall": round(wall_ms, 1),
            "response_chars": len(data.get("response", "")),
            "response_preview": (data.get("response") or "")[:120],
            "status": "ok",
            "error": None,
        }

        log.debug(
            "run_query: name=%s ok wall_ms=%.1f reported_ms=%s response_chars=%d",
            name, wall_ms, data.get("duration_ms"), result["response_chars"],
        )

    except Exception as exc:
        wall_ms = (time.perf_counter() - t0) * 1000
        result = {
            "scenario": name,
            "model_tier": scenario["model"],
            "tags": scenario["tags"],
            "prompt_chars": len(scenario["prompt"]),
            "system_chars": len(scenario.get("system") or ""),
            "max_tokens": scenario["max_tokens"],
            "model_reported": None,
            "duration_ms_reported": None,
            "duration_ms_wall": round(wall_ms, 1),
            "response_chars": 0,
            "response_preview": "",
            "status": "error",
            "error": str(exc),
        }

        log.error("run_query: name=%s exception type=%s msg=%s", name, type(exc).__name__, exc)

    return result

# -------------------------------------------------------------------------------------
# ------------------------------------ runner ----------------------------------------
# -------------------------------------------------------------------------------------

def print_row(result: dict) -> None:
    status = "✓" if result["status"] == "ok" else "✗"
    wall = result["duration_ms_wall"]
    reported = result["duration_ms_reported"]
    reported_str = f"{reported}ms" if reported is not None else "n/a"

    print(
        f"  {status} {result['scenario']:<45} "
        f"wall={wall:>7.1f}ms  api={reported_str:>8}  "
        f"chars_out={result['response_chars']:>5}"
    )

    if result["status"] == "error":
        print(f"      ✗ ERROR: {result['error']}")


def run(scenarios: list[dict]) -> list[dict]:
    log.debug("run: starting timing run scenario_count=%d", len(scenarios))
    print(f"\nRunning {len(scenarios)} scenario(s) against {QUERY_URL}\n")

    health = check_health()
    print(f"Health: {json.dumps(health)}\n")

    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario['name']} …", flush=True)
        result = run_query(scenario)
        print_row(result)
        results.append(result)

    return results


def save_results(results: list[dict]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"timing_{ts}.json"

    summary = {
        "run_at": ts,
        "base_url": BASE_URL,
        "scenario_count": len(results),
        "ok_count": sum(1 for r in results if r["status"] == "ok"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }

    out_path.write_text(json.dumps(summary, indent=2))
    log.debug("save_results: wrote %s", out_path)
    return out_path


def print_summary(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*60}")
    print(f"  Total: {len(results)}  ✓ {len(ok)}  ✗ {len(errors)}")

    if not ok:
        return

    by_tier: dict[str, list[float]] = {}
    for r in ok:
        tier = r["model_tier"]
        by_tier.setdefault(tier, []).append(r["duration_ms_wall"])

    print("\n  Wall-clock latency by model tier:")
    for tier, times in sorted(by_tier.items()):
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        print(f"    {tier:<10}  avg={avg:>7.0f}ms  min={mn:>7.0f}ms  max={mx:>7.0f}ms  n={len(times)}")

    by_tag: dict[str, list[float]] = {}
    for r in ok:
        for tag in r["tags"]:
            by_tag.setdefault(tag, []).append(r["duration_ms_wall"])

    print("\n  Wall-clock latency by tag:")
    for tag, times in sorted(by_tag.items()):
        avg = sum(times) / len(times)
        print(f"    {tag:<25}  avg={avg:>7.0f}ms  n={len(times)}")

# -------------------------------------------------------------------------------------
# ----------------------------------- entrypoint -------------------------------------
# -------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Time claude-code-api across scenarios")
    parser.add_argument("--tags", nargs="*", help="Only run scenarios with all of these tags")
    parser.add_argument("--model", help="Only run scenarios with this model tier (fast/balanced/best)")
    parser.add_argument("--names", nargs="*", help="Only run scenarios with these names")
    args = parser.parse_args()

    log.debug("main: args=%s", vars(args))

    scenarios = SCENARIOS

    if args.tags:
        scenarios = [s for s in scenarios if all(t in s["tags"] for t in args.tags)]
        log.debug("main: filtered by tags=%s remaining=%d", args.tags, len(scenarios))

    if args.model:
        scenarios = [s for s in scenarios if s["model"] == args.model]
        log.debug("main: filtered by model=%s remaining=%d", args.model, len(scenarios))

    if args.names:
        scenarios = [s for s in scenarios if s["name"] in args.names]
        log.debug("main: filtered by names=%s remaining=%d", args.names, len(scenarios))

    if not scenarios:
        print("✗ No scenarios matched the given filters.")
        sys.exit(1)

    results = run(scenarios)
    print_summary(results)
    out_path = save_results(results)
    print(f"\n  Results saved to: {out_path}")
    log.debug("main: done out_path=%s", out_path)


if __name__ == "__main__":
    main()
