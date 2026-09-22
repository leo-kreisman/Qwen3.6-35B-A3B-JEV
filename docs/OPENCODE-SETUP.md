# Running JEV SSD-offloading and driving it from OpenCode

Two separate paths, and they need different things from OpenCode. Pick one.

| | what it is | what OpenCode needs | can OpenCode do it today? |
|---|---|---|---|
| **A. Chat path** | `llama-server` on our 20.9 GB GGUF, `-ngl 0` | an OpenAI `/v1` endpoint | **yes**, nothing to build |
| **B. Decision path** | `systemone_shim.py`, one prefill per decision | the TypeSafe System One contract | provider is written; **no OpenCode adapter yet** |

Everything below was checked against the actual box — `llama-server` is built, OpenCode
1.18.31 is installed, and `/v1/models` and `/v1/systemone` both exist in the shim's own
route table. Nothing here is guessed.

---

## Path A — chat, and the one config change you need

### A0. The port already in the config is taken

`~/.config/opencode/opencode.jsonc` points at **`127.0.0.1:8077`**, and that model entry
is `qwen3.8-27b` — a different model. Right now nothing is listening on 8077; only
`11434` is up. Serve the 35B on its own port so you can have both:

**Use port 8090.** Leave `8077` for Qwen3.8-27B.

### A1. Start the server

```bash
cd ~/Projects/JEV_experiment

/home/scribe/Projects/llama.cpp-fresh/build/bin/llama-server \
  -m ~/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --alias qwen3.6-35b-a3b \
  -ngl 0 \
  -t 6 -ngl 0 -ub 2048 -b 2048 \
  -c 8192 \
  --jinja --reasoning-format deepseek \
  --temp 0.2 --top-p 0.9 --top-k 40 --repeat-penalty 1.05 \
  --host 127.0.0.1 --port 8090
```

Two flags matter more than the rest:

- **`-ngl 0`** — CPU only. This is the premise. Do not add `-ncmoe` and do not let a GPU
  get involved.
- **`-ub 2048 -b 2048`** — this is the measured 3.49× prefill lever. At the default 512
  the same server reads 135 GB to process 2,048 tokens; at 2048 it reads 37.81 GB.
  Measured cold under a verified 8 GiB cap: 29.09 → 101.59 prefill tok/s.

Also worth knowing:

- **`-t 6`** — 6 physical cores. The default is 16 (`nproc*4/3`), which measures
  **1.66–1.71× slower** (6.43 tok/s vs 10.94 in the thread sweep). Do not leave it default.
- **`--jinja --reasoning-format deepseek`** — preserve thinking. Non-negotiable for
  agentic coding with this model.
- **`-c 8192`, not 196608.** OpenCode resends the whole conversation every turn, and
  A3B models degrade badly past ~30k (Qwen3-Coder-30B-A3B, same class, drops to ~7%
  resolve at 64k). Keep the window small and start fresh sessions.
- **No `--api-key-file`.** The config uses `apiKey: "llama-cpp-local"`, which is a local
  placeholder — adding key enforcement would break it. This is localhost only; nothing
  leaves the box.

Wait for `server is listening on http://127.0.0.1:8090`, then confirm:

```bash
curl -s http://127.0.0.1:8090/v1/models | python3 -m json.tool | head -20
```

### A2. Add the model to OpenCode

This is the only file to edit: **`~/.config/opencode/opencode.jsonc`**. Inside
`"provider"`, next to the existing `local` and `ollama` blocks, add a third provider.
Leave the existing two alone.

```jsonc
"jev": {
  "name": "JEV SSD-offload (35B-A3B)",
  "options": {
    "baseURL": "http://127.0.0.1:8090/v1",
    "apiKey": "llama-cpp-local"
  },
  "models": {
    "qwen3.6-35b-a3b": {
      "id": "qwen3.6-35b-a3b",
      "name": "Qwen3.6-35B-A3B (SSD stream, CPU-only)",
      "reasoning": true,
      "tool_call": true,
      "cost": { "input": 0, "output": 0 },
      "limit": { "context": 8192, "output": 4096 }
    }
  }
}
```

`reasoning: true` is what keeps the thinking trace. `tool_call: true` is required for
OpenCode's tool use to work at all.

### A3. Run it

```bash
cd /path/to/your/project
opencode --model jev/qwen3.6-35b-a3b
```

Or set it permanently in `opencode.jsonc` with `"model": "jev/qwen3.6-35b-a3b"`.

To check the model is really the one answering:

```bash
opencode --model jev/qwen3.6-35b-a3b run "Say only: alive"
```

### A4. Measure it yourself

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b",
       "messages":[{"role":"user","content":"count to 20"}],
       "max_tokens":64,"stream":false}' \
| python3 -c "import json,sys; r=json.load(sys.stdin); t=r['timings']; \
print('prefill', round(t['prompt_per_second'],2), 'tok/s'); \
print('decode ', round(t['predicted_per_second'],2), 'tok/s')"
```

Expect roughly **decode ~10.7 tok/s warm**. That is the honest chat number. Prefill should
land far above the 203 tok/s the repo recorded at ub 512, because of the ubatch change.

### A5. Reality check on the chat path

**This is not the 20+ tok/s path.** Chat generates tokens autoregressively, so it pays
per-token cost — ~212.6 MB read per generated token cold. The measured decode numbers for
this model on this box are **3.67 tok/s cold under a binding cap, 10.68 warm**. The
ubatch fix does nothing for decode; it is a prefill lever.

Warm and uncapped it is genuinely usable. Cold and capped it is not fast. If the process
fits in your 62 GiB RAM with the checkpoint page-cached, you get the warm number —
that is the case the box is actually in, since the file is 20.9 GB and there is no cap
unless you impose one.

---

## Path B — decisions, the actual JEV path

This is the fast one: **one prefill pass per decision, `usage.output_tokens: 0`, cost
independent of how many options there are.**

### B1. Start the shim

```bash
cd ~/Projects/JEV_experiment

semif/.venv/bin/python resident/systemone_shim.py \
  --gguf ~/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --model ~/models/Qwen3.6-35B-A3B-tokenizer \
  --port 8123
```

It serves `POST /v1/systemone` and `GET /v1/models`. It takes the *same*
`--gguf/--model` pair as `resident_scorer.py` — the tokenizer directory is required, and
note it is `~/models/Qwen3.6-35B-A3B-tokenizer`, not the GGUF folder.

Point a client at it:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8123
```

### B2. The blocker — OpenCode cannot use it yet

**`/v1/systemone` is not an OpenAI chat-completions endpoint**, so pointing an OpenCode
`provider` at it will not work. There is no OpenCode adapter for the contract.

There are three ways across, in order of how much work they are:

1. **Bridge it with a tiny OpenAI-shaped wrapper** that translates a chat request into
   one System One call and returns the decision as a one-token completion. Small, but
   it is new code and it changes the token semantics — `usage.output_tokens` would no
   longer be 0.
2. **Register the shim as an MCP tool** so the agent calls `score_decision(...)` as a
   tool and reads back the probabilities. This is the honest shape: a decision is a
   tool call, not a chat turn.
3. **Have your application call `/v1/systemone` directly** and skip OpenCode entirely
   for the decision path.

**Recommendation: option 2.** It keeps the contract intact and matches how the repo
already frames this — `SETUP.md` says the shim exists so "the local SSD-served model
becomes a drop-in provider" for a caller that speaks the System One contract. OpenCode is
not that caller; the tool is the right seam.

**This is not built.** Do not read the section above as working code.

---

## What to actually do, in order

1. Start `llama-server` on **8090** with `-ngl 0 -t 6 -ub 2048 -b 2048 --jinja`.
2. Add the `jev` provider block to `~/.config/opencode/opencode.jsonc`.
3. `opencode --model jev/qwen3.6-35b-a3b` and confirm with `"Say only: alive"`.
4. Measure with the curl above; expect ~10.7 decode warm.
5. If you want the *fast* path, decide how OpenCode reaches `/v1/systemone` — the MCP
   tool route is the one that matches the design. That is a build, not a config.

**Do not** point OpenCode at `8077` — that is Qwen3.8-27B. **Do not** add a GPU flag.
**Do not** raise `-t` above 6.

---

## Known issues carried from the stack

- **Context bloat looks like a server slowdown.** A 35B that answers fast then crawls is
  usually a bloated session — every turn re-processes the whole history. Start a fresh
  session; do not extend a long one (`-c 8192` here keeps the ceiling honest).
- **Sampling belongs in both places.** `chat.params` in a plugin overrides the server
  per-request, but if the plugin fails to load the server flags are the fallback. Set both.
- **OpenSpec skills are auto-loaded by OpenCode** and the model may start a workflow on
  its own. If that bites, it is a plugin issue, not a model issue.
- **llama-server logs are CWD-dependent** if you redirect to a relative path. Always `cd`
  first or use an absolute log path.
