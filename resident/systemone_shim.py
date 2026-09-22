"""A local implementation of the TypeSafe System One HTTP contract.

JEV clients send a *state* and typed *questions* and read typed answers back;
they never generate text. This project's scorer already has that shape
(state + options -> one score per option), so this module is a **mapping**, not a
new engine. It reuses SemIf's verified path unchanged -- the reference-tokenizer
prompt render, the GGUF/HF agreement check behind ``prompt_sha256``, the
shared-state fan-out, and the option-slot readout -- so an answer here means the
same thing as an answer from ``cli.py``.

**Why this is the fast path, not the generated one.** A decision is a single
prefill pass whose cost does not grow with the number of options. Measured on this
host, cold and capped at 8 GiB: decode runs at **3.67 tok/s** and **212.6 MB per
generated token**. A generated answer would pay that per token; a decision does not.

**Why it exists at all.** TypeSafe documents ``base_url`` / ``TYPESAFE_BASE_URL``
as the supported extension point -- their own docs demo it against OpenRouter. So
pointing that variable at this process makes the local SSD-served model a drop-in
provider for ``typesafe-sdk``, Pydantic AI (``TypeSafeProvider``), ``jevgrep``,
and ``omp``'s ``api: typesafe`` provider, with no fork in any of them.

Contract
--------
``POST /v1/systemone``
    Request:  ``{"model": str, "state": str | object | [str], "questions": {...}}``
    Response: ``{"model": str, "answers": {...}, "usage": {"input_tokens": int,
              "output_tokens": int}}`` plus a non-conflicting ``local`` telemetry key.

Question mapping -- the only place that needs care, and it is where this differs
from a real Jev:

    ``noul``    two options (holds / does not hold); the answer is p(holds).
                Real Jev resolves a noul in one head; here it is one binary
                comparison, so treat the value as a decision probability, not a
                calibrated one. See ``probability_status`` on the scorer output.
    ``choice``  one option per key in ``criteria``; probabilities over the caller's keys.
    ``score``   one option per level in the ordered ``criteria`` array; the score is
                the probability-weighted mean of the level indices, so a 4-level
                scale returns 0.0-3.0.

Every string returned is the caller's own text echoed back. Nothing is generated.

Usage
-----
    python systemone_shim.py --gguf <path> --model <tokenizer-dir> [--port 8123]
    export TYPESAFE_BASE_URL=http://127.0.0.1:8123
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resident_scorer import ResidentScorer  # noqa: E402

LOCAL_MODEL = "local-qwen3.6-35b-a3b-ssd"
MAX_CHOICE_OPTIONS = 255          # the contract's limit
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10

# Real Jev resolves a noul as a single calibrated head. A binary comparison is the
# closest thing this scorer can express, and the wording matters because these are
# the strings the model actually reads.
NOUL_OPTIONS = [
    {"id": "holds",
     "description": "Yes. The statement above is true of the state described."},
    {"id": "does_not_hold",
     "description": "No. The statement above is not true of the state described."},
]


class InvalidRequest(ValueError):
    """A request that should be reported as 422 invalid_request."""


def _state_text(state) -> str:
    """The contract accepts a string, an object, or an array of strings."""
    if isinstance(state, str):
        if not state.strip():
            raise InvalidRequest("`state` must not be empty")
        return state
    if isinstance(state, dict):
        if not state:
            raise InvalidRequest("`state` object must not be empty")
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    if isinstance(state, list):
        if not state or not all(isinstance(item, str) for item in state):
            raise InvalidRequest("`state` array must be a non-empty list of strings")
        return "\n\n".join(state)
    raise InvalidRequest("`state` must be a string, an object, or an array of strings")


def _score_levels(criteria) -> list[str]:
    """The contract orders score levels as an array; numeric-keyed maps appear in
    the wild, so accept both but always resolve to an index-ordered list."""
    if isinstance(criteria, dict):
        try:
            ordered = sorted(criteria.items(), key=lambda item: float(item[0]))
        except (TypeError, ValueError) as error:
            raise InvalidRequest("score `criteria` keys must be numeric") from error
        levels = [description for _, description in ordered]
    elif isinstance(criteria, list):
        levels = criteria
    else:
        raise InvalidRequest("score `criteria` must be an array of level descriptions")
    if not all(isinstance(level, str) and level.strip() for level in levels):
        raise InvalidRequest("score levels must be non-empty strings")
    if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
        raise InvalidRequest(
            f"score needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, got {len(levels)}"
        )
    return levels


def build_rows(state: str, questions: dict) -> tuple[list[dict], dict]:
    """One scorer row per question, plus the layout needed to map answers back."""
    rows: list[dict] = []
    layout: dict[str, dict] = {}

    for question_id, question in questions.items():
        if not isinstance(question, dict):
            raise InvalidRequest(f"question {question_id!r} must be an object")
        kind = question.get("type")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise InvalidRequest(f"question {question_id!r} needs non-empty `instructions`")

        if kind == "noul":
            options, text = NOUL_OPTIONS, instructions
        elif kind == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise InvalidRequest(f"choice question {question_id!r} needs a `criteria` object")
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise InvalidRequest(
                    f"choice question {question_id!r} has {len(criteria)} options, "
                    f"the limit is {MAX_CHOICE_OPTIONS}"
                )
            options, text = (
                [{"id": key, "description": str(value)} for key, value in criteria.items()],
                instructions,
            )
        elif kind == "score":
            levels = _score_levels(question.get("criteria"))
            options, text = (
                [{"id": str(index), "description": level} for index, level in enumerate(levels)],
                instructions,
            )
        else:
            raise InvalidRequest(f"question {question_id!r} has unsupported type {kind!r}")

        rows.append({"id": question_id, "state": state, "question": text, "options": options})
        layout[question_id] = {"type": kind, "option_ids": [o["id"] for o in options]}
    return rows, layout


def build_answers(results: list[dict], layout: dict) -> dict:
    """Map scorer output onto the contract's answer objects, one shape per type."""
    answers: dict[str, dict] = {}
    for result in results:
        slot = layout.get(result["id"])
        if slot is None:
            continue
        probabilities = {
            option_id: float(probability)
            for option_id, probability in zip(result["option_ids"], result["probabilities"])
        }
        best = max(probabilities, key=probabilities.get)
        kind = slot["type"]

        if kind == "noul":
            # The contract has no separate confidence here: the probability is the
            # certainty measure, and 0.5 is maximal uncertainty.
            answers[result["id"]] = {"type": "noul", "noul": probabilities["holds"]}
        elif kind == "choice":
            answers[result["id"]] = {
                "type": "choice",
                "choice": best,
                "confidence": probabilities[best],
                "probabilities": probabilities,
            }
        else:
            answers[result["id"]] = {
                "type": "score",
                "score": sum(index * probabilities[option_id]
                             for index, option_id in enumerate(slot["option_ids"])),
                "confidence": probabilities[best],
                "legend": {option_id: description for option_id, description in
                           zip(slot["option_ids"], slot["option_descriptions"])},
                "probabilities": probabilities,
            }
    return answers


class ShimHandler(BaseHTTPRequestHandler):
    scorer: ResidentScorer = None       # set once in main()
    call_lock = threading.Lock()        # llama.cpp contexts are not re-entrant
    protocol_version = "HTTP/1.1"
    server_version = "systemone-shim/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("shim: " + fmt % args + "\n")

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _invalid(self, message: str) -> None:
        self._send(422, {"error_type": "invalid_request", "error": message})

    def do_GET(self):  # noqa: N802
        if self.path in ("/health", "/v1/health"):
            self._send(200, {
                "status": "ok",
                "model": LOCAL_MODEL,
                "load_seconds": round(self.scorer.load_seconds, 3),
                "calls": self.scorer.calls,
            })
        elif self.path == "/v1/models":
            self._send(200, {"models": [LOCAL_MODEL, "jev-latest", "jev-preview"]})
        else:
            self._send(404, {"error_type": "not_found", "error": "no such route"})

    def do_POST(self):  # noqa: N802
        if self.path not in ("/v1/systemone", "/systemone"):
            self._send(404, {"error_type": "not_found", "error": "no such route"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise InvalidRequest("expected a JSON body")
            request = json.loads(self.rfile.read(length))
        except InvalidRequest as error:
            self._invalid(str(error))
            return
        except (ValueError, TypeError) as error:
            self._invalid(f"malformed JSON body: {error}")
            return

        if not isinstance(request, dict):
            self._invalid("request body must be a JSON object")
            return

        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            self._invalid("`questions` must be a non-empty object")
            return

        try:
            state = _state_text(request.get("state"))
            rows, layout = build_rows(state, questions)
        except InvalidRequest as error:
            self._invalid(str(error))
            return

        started = time.perf_counter()
        try:
            with self.call_lock:
                results, report = self.scorer.call(rows)
        except Exception as error:  # noqa: BLE001 - decode failure must surface as 5xx
            self._send(500, {"error_type": "scoring_failed", "error": repr(error)})
            return

        # Keep the descriptions available for `legend` without re-deriving them.
        for row in rows:
            layout[row["id"]]["option_descriptions"] = [o["description"] for o in row["options"]]

        self._send(200, {
            "model": request.get("model") or LOCAL_MODEL,
            "answers": build_answers(results, layout),
            "usage": {
                "input_tokens": sum(r["input_tokens"] for r in results),
                "output_tokens": 0,
            },
            "local": {
                "model": LOCAL_MODEL,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "call_read_bytes": report.get("call_read_bytes", 0),
                "process_read_bytes": report.get("process_read_bytes", 0),
                "output_tokens": 0,
                # read_bytes == 0 is the tell of a page-cache-warm run, which is a fact
                # about the measurement -- not a broken counter. Name it, so a reader
                # cannot mistake one for the other. See docs/SEMIF_LLAMACPP_SSD.md.
                "cache_state": ("warm - served from page cache, no device reads"
                                if not report.get("call_read_bytes")
                                else "cold - served from NVMe"),
                "note": "no tokens are generated; `output_tokens` is 0 by construction",
            },
        })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Reference tokenizer directory")
    parser.add_argument("--revision", default="local")
    parser.add_argument("--llama-threads", type=int, help="default: 4/3 of nproc, the measured knee")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    ShimHandler.scorer = ResidentScorer(argparse.Namespace(
        gguf=args.gguf, model=args.model, revision=args.revision,
        llama_threads=args.llama_threads, max_tokens=args.max_tokens,
    ))

    server = ThreadingHTTPServer((args.host, args.port), ShimHandler)
    server.daemon_threads = True
    print(json.dumps({
        "systemone_shim": {
            "endpoint": f"http://{args.host}:{args.port}/v1/systemone",
            "models": f"http://{args.host}:{args.port}/v1/models",
            "load_seconds": round(ShimHandler.scorer.load_seconds, 3),
            "concurrency": "serialised - one llama.cpp context, one call at a time",
            "note": "export TYPESAFE_BASE_URL to this origin",
        }
    }), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
