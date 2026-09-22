# Third-party notices

This repository contains measurements, documentation, and a patch. It does not
vendor any third-party source tree. The components it depends on or modifies are
listed here, with what this repository does and does not claim over each.

## SemIf — patched, not redistributed

The scorer is **SemIf** by TheoLeeCJ, MIT licensed:

    https://github.com/TheoLeeCJ/SemIf
    LICENSE: MIT License, Copyright (c) 2026 TheoLeeCJ
    pinned at commit 1f2dea3e25379f9dfc98cb83c324f00ab5deda37

SemIf is **not** included in this repository. `patch/` carries a diff against
that pinned commit plus the three complete post-patch files. Both are derived
works of SemIf and remain under SemIf's MIT license; the MIT notice above
applies to them, and the patch preserves SemIf's copyright headers.

The three files this repository changes:

| File | Change |
| --- | --- |
| `src/semif_phase1/llamacpp_backend.py` | the substantive work — batched branch scoring, ubatch sizing, thread policy, digest cache |
| `tests/test_llamacpp.py` | added coverage for the above |
| `tests/test_llamacpp_batched.py` | new file, the batched-path regression suite |

Everything else in this repository (the documentation, the runner, the probes,
the results) is original to this project and covered by `LICENSE`.

## llama.cpp — depended on, not modified

`llamacpp_backend.py` drives llama.cpp through `ctypes`. It does not patch or
redistribute llama.cpp; it calls the shared library that SemIf's build provides.
Several findings here are *about* llama.cpp behaviour (`n_ubatch` deciding I/O,
`llama_memory_hybrid::seq_cp` aborting on this architecture, the mmap load path
touching the whole mapping) and are recorded as observations with the version
they were observed on, not as patches.

llama.cpp is MIT licensed: https://github.com/ggml-org/llama.cpp

## Qwen3.6-35B-A3B — downloaded, not redistributed

The model weights are not in this repository and are not covered by its license.
`SETUP.md` records the exact artifact measured:

    unsloth/Qwen3.6-35B-A3B-GGUF -> Qwen3.6-35B-A3B-UD-Q4_K_S.gguf

Model weights carry their own license from their publisher. Check it before
redistributing anything derived from them. The `results/*.jsonl` files here
contain only decision probabilities and token counts — no model weights, and no
model output beyond a label and a probability.

The tokenizer is fetched from `Qwen/Qwen3.6-35B-A3B` on Hugging Face by the
runner on first use, and is likewise governed by its own license.

## The twelve surveyed engines — surveyed, not redistributed

`docs/OFFLOAD_PROJECTS_ANALYSIS.md` is a survey of twelve existing
MoE-offload/SSD-streaming engines. Their source trees are excluded from this
repository (`offload_projects/` in `.gitignore`); `scripts/fetch_prior_art.sh`
re-clones each one at the commit that was read, so the survey can be re-checked
without this repository redistributing anyone's code. Each remains under its own
license.

Findings *about* those engines — which optimisations are measured dead ends,
which seam is the proven one — are this project's own work and are covered by
`LICENSE`.
