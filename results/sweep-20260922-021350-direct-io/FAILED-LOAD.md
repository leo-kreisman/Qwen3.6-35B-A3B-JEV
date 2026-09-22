# `LOAD_MODE=direct_io` cannot load under an 8 GiB cap — negative result

Command:

```sh
OUTDIR=results/sweep-20260922-021350-direct-io REPEAT=1 SEQ_MAX=8 LOAD_MODE=direct_io \
  ./run_resident_35b_ssd.sh --simulate-8g examples/decisions.sample.jsonl
```

Outcome: **SIGKILLed during load, exit code 137, before a single ledger line.** The
process never reached `resident_load`, so there is no `call*.jsonl` here — the
`run.log` is gitignored (`*.log`), which is why this note exists.

The evidence, from `run.log`:

```
load_tensors:          CPU model buffer size = 19914.65 MiB
load_all_data: no device found for buffer type CPU for async uploads
.....................................
./run_resident_35b_ssd.sh: line 111: 1777926 Killed   SEMIF_LLAMA_SEQ_MAX=... \
  "$PY" .../resident_scorer.py --gguf ... --llama-cpp ... 
```

`19914.65 MiB` is 20.88 GB — the whole checkpoint, allocated as anonymous RAM.
That is the mechanism: `direct_io` opens the GGUF with `O_DIRECT` and **reads every
tensor into allocated buffers instead of mapping them** (`llama-mmap.cpp:199`), so
the model must be *resident*, not streamed.

## Why this closes the lever rather than merely ranking it low

`_LOAD_MODES` exposes `direct_io` as value 4, and `llama_file::has_direct_io` is
exported in this build, so it is real and reachable. Two earlier sessions listed it
as untested and guessed at the tradeoff ("a faster load and a worse footprint").
Measured, there is no tradeoff to weigh: at the 8 GiB cap this project exists to
target, it does not load at all.

It is a load mode for a host that can hold the model resident. That is precisely
the host this work is *not* for — `run_semif_35b_ssd.sh --simulate-8g` exists to
model the machine that cannot.

## What this does not say

It does not say `O_DIRECT` is useless for SSDs. It says `O_DIRECT`-into-buffers is
the wrong shape here. The lever that could still work is `pread` over exact expert
extents with the model *not* resident and the engine owning a bounded slot pool —
which is a decode rewrite, not a load mode, and is scoped in `resident/README.md`.
