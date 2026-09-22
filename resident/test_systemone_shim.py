"""End-to-end test for the System One shim, against the real checkpoint.

Starts the shim, waits for it to load, sends one request exercising all three
question types, and asserts the response matches the contract shape. Exits
non-zero on any mismatch, so it is usable as a gate.

    python resident/test_systemone_shim.py --gguf <path> --model <tokenizer-dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

STATE = (
    "Page: Google Flights (one-way search form). Observed elements: [0] input 'Where from?' "
    "value 'Zürich'; [1] input 'Where to?' value ''; [2] button 'Departure' value 'Sat, Oct 18'; "
    "[3] button 'Search'; [4] link 'One way' state 'selected'. Goal: one-way Zürich to London "
    "on 2026-10-20."
)

REQUEST = {
    "model": "jev-latest",
    "state": STATE,
    "questions": {
        "destination": {
            "type": "choice",
            "instructions": "Which observed element should receive the destination value 'London'?",
            "criteria": {
                "e0": "Element 0, the origin input, which already holds Zürich.",
                "e1": "Element 1, the destination input, which is currently empty.",
                "e2": "Element 2, the departure date control.",
                "e3": "Element 3, the search control.",
            },
        },
        "confidence": {
            "type": "score",
            "instructions": "How clear is the correct element to fill?",
            "criteria": ["guesswork", "ambiguous", "likely", "certain"],
        },
        "needs_human": {
            "type": "noul",
            "instructions": "Should a human confirm this action before it is taken?",
        },
    },
}

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        failures.append(message)


def post(url: str, payload: dict, timeout: float = 300.0):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--llama-threads", type=int, default=16)
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    # The checkpoint load prints ~800 lines of llama.cpp loader output. If that
    # goes to a pipe nobody drains until the process exits, it fills the 64 KiB
    # buffer and the shim blocks on write *before* it ever binds the socket --
    # a deadlock that looks exactly like "never became healthy". Send it to a
    # file instead and tail it only when something goes wrong.
    log_path = HERE / f"systemone_shim.{args.port}.log"
    log = log_path.open("w")
    server = subprocess.Popen(
        [sys.executable, str(HERE / "systemone_shim.py"),
         "--gguf", str(args.gguf), "--model", args.model,
         "--port", str(args.port), "--llama-threads", str(args.llama_threads)],
        stdout=log, stderr=subprocess.STDOUT, text=True,
    )

    def tail_log(lines: int = 25) -> str:
        log.flush()
        try:
            content = log_path.read_text().splitlines()
        except OSError:
            return "(no log)"
        return "\n".join(content[-lines:]) or "(log empty)"

    try:
        print("waiting for the shim to load the checkpoint...")
        deadline = time.time() + 600
        ready = False
        while time.time() < deadline:
            if server.poll() is not None:
                print(f"shim exited early (rc={server.returncode}), last lines of "
                      f"{log_path}:\n{tail_log()}")
                return 1
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                    print("health:", json.loads(response.read()))
                    ready = True
                    break
            except Exception:
                time.sleep(2)
        if not ready:
            print(f"shim never became healthy after 600s; it is still running "
                  f"(rc={server.poll()}), last lines of {log_path}:\n{tail_log()}")
            return 1

        print("\n--- GET /v1/models ---")
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
            models = json.loads(response.read())
        print(models)
        check("models" in models and models["models"], "/v1/models returns a list")

        print("\n--- POST /v1/systemone (all three question types) ---")
        started = time.time()
        status, body = post(f"{base}/v1/systemone", REQUEST)
        elapsed = time.time() - started
        print(json.dumps(body, indent=2)[:2200])
        print(f"\nwall: {elapsed:.1f}s  status: {status}")

        check(status == 200, "200 OK")
        check(body.get("model") == "jev-latest", "model echoed")
        answers = body.get("answers", {})
        check(set(answers) == {"destination", "confidence", "needs_human"},
              "one answer per question")

        choice = answers.get("destination", {})
        check(choice.get("type") == "choice", "choice: type echoed")
        check(choice.get("choice") in REQUEST["questions"]["destination"]["criteria"],
              f"choice: winner is a caller-supplied key ({choice.get('choice')!r})")
        check(isinstance(choice.get("confidence"), float), "choice: confidence is a float")
        check(abs(sum(choice.get("probabilities", {}).values()) - 1.0) < 1e-4,
              "choice: probabilities sum to 1")
        check(choice.get("choice") == "e1",
              "choice: picks the empty destination input (semantic check)")

        score = answers.get("confidence", {})
        check(score.get("type") == "score", "score: type echoed")
        check(0.0 <= score.get("score", -1) <= 3.0,
              f"score: within 0.0-3.0 for a 4-level scale ({score.get('score'):.3f})")
        check(set(score.get("legend", {})) == {"0", "1", "2", "3"},
              "score: legend covers every level")
        check(score.get("legend", {}).get("3") == "certain",
              "score: legend echoes the caller's level text verbatim")

        noul = answers.get("needs_human", {})
        check(noul.get("type") == "noul", "noul: type echoed")
        check(isinstance(noul.get("noul"), float) and 0.0 <= noul["noul"] <= 1.0,
              f"noul: float in [0,1] ({noul.get('noul')})")
        check("confidence" not in noul and "probabilities" not in noul,
              "noul: no confidence/probabilities, per the contract")

        usage = body.get("usage", {})
        check(usage.get("input_tokens", 0) > 0, "usage: input_tokens counted")
        check(usage.get("output_tokens") == 0, "usage: output_tokens is 0 (nothing generated)")
        check(body.get("local", {}).get("note"), "local: telemetry present")
        check(body.get("local", {}).get("cache_state", "").split(" -")[0] in ("warm", "cold"),
              f"local: cache_state named ({body.get('local', {}).get('cache_state')!r})")

        print("\n--- POST /v1/systemone (validation) ---")
        for label, bad, expected in [
            ("empty questions", {"state": "x", "questions": {}}, 422),
            ("missing state", {"state": "", "questions": {"q": {"type": "noul",
                                                               "instructions": "x"}}}, 422),
            ("bad type", {"state": "x", "questions": {"q": {"type": "vibes",
                                                            "instructions": "x"}}}, 422),
            ("1-level score", {"state": "x", "questions": {"q": {"type": "score",
                                                                 "instructions": "x",
                                                                 "criteria": ["only"]}}}, 422),
        ]:
            status, payload = post(f"{base}/v1/systemone", bad)
            check(status == expected,
                  f"{label} -> {expected} ({status}, {payload.get('error_type')})")

        print("\n--- POST /v1/systemone (state as object and as array) ---")
        for label, state in [("object", {"page": "flights", "goal": "one-way"}),
                             ("array", ["Page: flights form.", "Goal: one-way Zürich to London."])]:
            status, payload = post(f"{base}/v1/systemone",
                                   {"state": state, "questions": REQUEST["questions"]})
            check(status == 200 and len(payload.get("answers", {})) == 3,
                  f"state as {label} accepted")

    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=20)
        log.close()

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
