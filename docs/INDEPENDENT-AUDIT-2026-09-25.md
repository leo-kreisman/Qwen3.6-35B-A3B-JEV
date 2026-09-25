# Independent assessment — 2026-09-25

The project has demonstrated important implementation limits, but has not established
a universal performance ceiling. A large speedup from simply repacking the same weights
is unlikely. Faster JEV decisions through less repeated work remain a credible research
direction. Twenty generated tokens/second on this CPU, with this checkpoint, under
8 GiB and unchanged model behavior, is not demonstrated or promised.

This assessment inspected source, recorded results, and the September 23 and September
25 Hermes session exports in `/home/scribe/.hermes/sessions/saved/`. It recalculated
arithmetic from JSON artifacts, but did not rerun model benchmarks. Existing reports
are historical evidence, not independent reproductions. No runtime code was changed.
Cross-field sources below were checked on the web.

## What the project currently does

- `semif/src/semif_phase1/llamacpp_backend.py`: CPU llama.cpp backend for typed
  decisions. Prompts include candidate labels; the model returns their conditional
  probabilities, without autoregressive generation.
- `resident/resident_scorer.py`: keeps the model/context alive across calls.
- `resident/systemone_shim.py`: exposes the typed `/v1/systemone` contract.
- `patch/`: distributable SemIf modifications. `semif/`, `vendor/`, and external
  llama.cpp builds mean that this is not one self-contained runtime artifact.
- `vendor/BigMoeOnEdge/`: a separate expert-streaming generation path. Its good
  generation results do not automatically improve the SemIf scorer.
- `scripts/io/`: expert-pack and alignment experiments; a fast storage benchmark
  does not establish that the scorer consumes the new format.

The working premise inherited from the project is CPU-only, a binding 8 GiB total
memory budget, and quality-preserving changes first. The host itself has more RAM;
the cap is an experimental/deployment constraint.

## Evidence and verdicts

| Claim | Assessment |
|---|---|
| Ordinary CPU decode is strongly bandwidth constrained | Plausible and supported by warm decode and thread sweeps; exact DRAM traffic was not established by the tensor inventory alone. |
| 15.5 tok/s is an absolute hardware limit | Too strong. It divides a reported microbenchmark result by estimated weight traffic; neither is an unconditional bound on all algorithms. |
| The same bound applies to JEV decision throughput | Incorrect. JEV is batched prefill with one readout per decision. |
| Alignment/repacking alone solves generation | Unsupported. The recorded aligned run moved 3.699 to 3.787 tok/s, only 2.38%, over 32 generated tokens. |
| The source GGUF cannot be read using O_DIRECT | Too broad. Unaligned direct requests fail, but aligned enclosing reads into bounce buffers work. The current streaming reader supports this approach. |
| Generic lossless compression will make this checkpoint several times smaller | No supporting measurements. The sampled weighted zstd artifact implies only 4.76% saving for the routed set. |
| Those compression tests prove that all lossless encodings are exhausted | Incorrect. Sampled compressor ratios and marginal byte entropy are not a proof about all structured encodings. |
| Speculation automatically doubles speed | Incorrect. Accepted tokens, expert-union growth, rollback memory and verification cost determine the result. |
| All speculation is disproven | Also too strong. The vendor n-gram report is negative, but the same report describes a host MTP gain on another checkpoint/configuration. Neither transfers automatically here. |
| The resident scorer still needs to be built | Stale: it exists. Integrating another engine is a separate task. |

### Recomputed decode arithmetic

From `results/gguf-byte-breakdown.json`:

```
routed expert pool                         18.327011328 GB
always-active classes, excluding embedding  2.014669312 GB
estimated active weights, top-8 of 256       2.587388416 GB/token
required weight-read rate at 20 tok/s       51.747768320 GB/s
40.22 GB/s / estimated active weights      15.5446 tok/s
```

These are a useful first-order model of ordinary single-token decode. They omit
activation traffic and may differ from actual memory-controller traffic because of
cache reuse, implementation-specific tensor access, writes and temporary buffers.
The session's 40.22 GB/s is an observed bandwidth result, not a proof that no other
access pattern can exceed it. The session also acknowledges that DDR4-2400 was not
verified. A small-region L3 test is not a universal L3 bandwidth bound either.

Likewise, an instrumented `compute` interval includes memory stalls inside kernels.
Treating its 0.09 s/token as bandwidth-independent ALU time would double-count the
same bottleneck. Multiplying token rate by an assumed byte count gives an estimated
bandwidth; it does not independently validate that byte count.

This interpretation follows the original [Roofline paper](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/Archive/EECS-2008-134.pdf):
the relevant traffic is between DRAM and caches, and changing data reuse changes
operational intensity.

### Results that survive inspection

- `results/bmoe-sweep-20260922-191242/sweep.jsonl`: 3.834 tok/s without expert
  cache; 7.408 with 4000 MiB cache and eight I/O lanes. This is a combined
  configuration comparison, not an isolated cache-policy comparison.
- `results/pipeline/ab-124355.jsonl`: roughly 10.72 to 12.79 warm decode tok/s
  after non-expert requantization. This changes numerics; the recorded perplexity
  check is not a broad task-quality guarantee.
- `results/probe-readamp-20260922-174920/configs.jsonl`: prefill 29.09 to 101.59
  tok/s as ubatch grows 512 to 2048; estimated inference reads 135.08 to 37.81 GB.
  Cap peaks are recorded at 8 GiB. Read attribution subtracts a separate load-only
  baseline, so the residual is not a per-tensor trace.
- `results/aligned-e2e-20260923-034709/`: source and aligned generation both
  already report direct I/O enabled; approximately 0.094/0.093 s compute and
  0.176/0.171 s I/O per token. This explains why alignment alone changed little.
- `results/q4ks-weighted.json`: sampling is 384 KiB per tensor across 120 expert
  tensors, weighted by size. It reports 95.24% of raw size for the routed set.
  Some prose says 5.11% saving; use the artifact's scope and number when comparing.

Two documentation mistakes matter for choosing the next experiment:

1. The current scorer `_ubatch(sequences)` defaults to `max(512, sequences*512)`,
   not a fixed 512. Raising it to 2048 is not a newly discovered universal fix.
   The historical scorer probe already found that increasing an adequate ubatch
   did not eliminate its residual reads.
2. `docs/CEILING-CPU-ONLY.md` mixes an older 1.989 GB/token approximation with later
   tensor-based 2.587 GB/token figures. Its quantization table is also arithmetically
   inconsistent: 57% of 38.4 GB/s divided by 1.989 GB is about 11, not 16.4 tok/s.

## Cross-field directions with a concrete connection to this code

### 1. Numerical computing: reuse weights across ready work

[ATLAS](https://www.netlib.org/atlas/developer/atlas_contrib/node10.html) uses blocked
matrix multiplication to reuse data in cache. The transferable idea is to organize
execution around a loaded weight block, processing all ready rows that need it.

For JEV, investigate layer/expert scheduling across prompt chunks and independent
criteria, with explicit buffers. Reuse each expert before eviction rather than walking
the entire model separately for every microbatch. The recorded 3.49x prefill gain
shows why this direction matters, but does not measure this proposed scheduler.

This is a substantial engine change. Qwen's hybrid attention/recurrent dependencies
must be preserved. Available rows, causal order, recurrent state, activation storage
and expert accumulation order constrain legal reordering. Small activation buffers
may be cheaper to spill than reloading expert matrices, but that needs a byte budget
and a prototype. A different reduction order can change floating-point results.

### 2. Databases: compute only the requested result

[DuckDB's Parquet projection pushdown](https://duckdb.org/docs/stable/data/parquet/overview)
reads only requested columns. The analogous JEV opportunity is an answer-only output
projection: compute the rows of the language-model head for the declared answer
tokens, rather than every vocabulary token.

The current code computes full logits, then indexes `vocabulary[slots]`.
`_result()` applies softmax to just these selected logits. With final hidden state
`h`, answer probabilities require only:

```
z_i = W_output[i] h                  for i in allowed answer slots
p_i = exp(z_i) / sum_j exp(z_j)      over those slots
```

The full vocabulary has 248,320 entries. The recorded output-projection class is
about 417 MB; 255 rows would be approximately 0.43 MB of weights, before alignment
and ancillary data. This is a reduction of this operation, not a 1000x model speedup.
The head is evaluated at scored positions, not at every prefill input token, so its
share must be profiled rather than borrowed from the decode 16.1% share.

Important contract restriction: `allowed_token_mass` and `full_vocab_argmax_id` need
the full vocabulary. They cannot be preserved by answer-only projection. The typed
shim's `build_answers()` uses conditional option probabilities, so a dedicated
decision-only mode could retain that answer contract while keeping full diagnostics
in the original mode. Verify quantized row kernels and near-tie decisions; changing
matrix dimensions can change numerical accumulation.

### 3. Database buffer management: protect reusable data from scans

[IBM's ARC](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache)
is scan-resistant and balances recency against frequency. The relevant question is
whether prefill scans are evicting experts or dense data needed soon afterwards.

Capture accesses by layer/expert/byte size and replay LRU, scan-bypass, frequency and
cost-aware policies under the same byte budget. Use an offline optimum only as a
reference; it cannot predict future routed experts in production. These experts are
not all equal sized, so a uniform-page algorithm needs adaptation. A high existing
decode cache hit rate is not evidence that an alternative policy will help.

### 4. Scientific I/O: plan reads collectively

[ROMIO](https://ftp.mcs.anl.gov/pub/romio/users-guide/node5.html) combines noncontiguous
requests and coordinates reads. Apply the principle to a known routed set: deduplicate
requests, sort by file region, coalesce nearby ranges and stage into bounded buffers.
Choose coalescing by a measured latency-versus-extra-bytes cost, not merely minimum
request count. Reading across hundreds of MB of unused data would be counterproductive.

The pack benchmark supports this direction for isolated I/O. It does not establish
end-to-end improvement, and the alignment result warns against assuming one.

### 5. Incremental computation: avoid identical prefix work

`score_shared()` currently replicates the full state prefix in every sequence. The
serial path can save/restore state, but batching avoids repeated model sweeps under
the cap. The desired combination is explicit shared prefix state followed by batched
suffixes, including correct copies of recurrent state. The current hybrid `seq_cp`
limitation prevents treating this as a flag change. Exact request/result caching is
another option only when identical inputs actually recur; include model, tokenizer,
prompt template and scoring configuration in the key.

## A bounded experimental order

1. Establish one current scorer baseline and one generation baseline separately.
   Record exact model hash, executable/library provenance, cap peak, cache state,
   useful decisions or accepted tokens, and wall latency. Profile per-layer compute,
   expert unique bytes, repeated reads, wait time and output-head time. Do not count
   summed concurrent I/O worker time as serialized wall time.
2. For JEV, attribute the remaining ~38–41 GB/pass before rewriting the loader.
   If repeated expert reads dominate, prototype expert/row scheduling. If the final
   head is material, prototype answer-only projection. Measure complete call latency
   and compare outputs on varied, ambiguous and near-tie fixtures.
3. Replay cache policies from the trace before adding cache machinery. Reject a
   policy whose byte misses and estimated exposed stalls do not improve.
4. Integrate packed reads only if traced I/O remains a material part of the critical
   path. Use alternating repeated A/B runs under the same verified cap.
5. Treat generation speculation as a separate project. Its success condition is
   accepted tokens per total draft/verification/I/O/rollback time, not draft speed.
   Do not repeat the existing negative n-gram tuning without a changed hypothesis.

There is enough evidence to stop promising a magic file-format solution. There is
not enough evidence to stop investigating fewer passes, selective computation and
better reuse for the actual typed-decision workload.
