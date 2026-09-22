"""CPU option readout over GGUF checkpoints through llama.cpp.

Prompt construction and answer-slot verification stay on the reference
transformers tokenizer, so prompt_sha256 matches the Torch backend exactly;
llama.cpp only executes the forward pass over the quantized GGUF weights.
Every scored prompt is re-tokenized through the GGUF vocabulary and must
agree with the reference encoding before it is evaluated.

Sequence 0 of the single context holds the prefill; each decision restores the
saved prefix state (whole-sequence save/restore) before decoding its suffix.
The hybrid linear-attention memory of Qwen3.5 supports neither sequence
copies nor partial tail removal, so branch replication goes through the
per-sequence state serialization llama.cpp itself uses for slot caching.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time
import weakref

import numpy

from .core import LETTERS, direct_messages, softmax
from .direct import PROMPT_VERSION, encode_prompt
from .shared import _state_prefix

DECODE_CHUNK = 512
_BACKEND_INITIALIZED = False


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment override; unset keeps the default."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


_LOAD_MODES = {"auto": -1, "none": 0, "mmap": 1, "mlock": 2, "mmap_mlock": 3, "direct_io": 4}
_LOAD_MODE_NAMES = {value: name for name, value in _LOAD_MODES.items()}


def _load_mode() -> int:
    """Resolve ``SEMIF_LLAMA_LOAD_MODE`` to a ``llama_load_mode`` value."""
    name = os.environ.get("SEMIF_LLAMA_LOAD_MODE", "mmap").strip().lower()
    if name not in _LOAD_MODES:
        raise ValueError(
            f"unknown SEMIF_LLAMA_LOAD_MODE {name!r}; expected one of {sorted(_LOAD_MODES)}"
        )
    return _LOAD_MODES[name]


def _seq_max() -> int:
    """Resolve ``SEMIF_LLAMA_SEQ_MAX`` to the context's branch capacity.

    This is how many criteria can be scored in a *single* forward pass. It is a
    context-creation parameter (``n_seq_max``), so changing it means the model
    must be reloaded; the default covers the usual JEV batch without making every
    context pay for a large KV.
    """
    raw = os.environ.get("SEMIF_LLAMA_SEQ_MAX", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"SEMIF_LLAMA_SEQ_MAX {raw!r} is not an integer") from None
    if value < 1:
        raise ValueError(f"SEMIF_LLAMA_SEQ_MAX must be at least 1, got {value}")
    return value


def _ubatch(sequences: int) -> int:
    """Physical batch size: enough to hold one whole branch group in one ubatch.

    Every ubatch walks all layers and re-reads their experts, so a group split
    across two ubatches pays two full model sweeps — and llama.cpp's default of
    512 did exactly that to a 908-token four-criterion group without saying so.
    A group is at most one sequence per criterion, so size by ``sequences``.
    """
    pinned = os.environ.get("SEMIF_LLAMA_UBATCH", "").strip()
    if pinned:
        try:
            value = int(pinned)
        except ValueError:
            raise ValueError(f"SEMIF_LLAMA_UBATCH {pinned!r} is not an integer") from None
        if value < 1:
            raise ValueError(f"SEMIF_LLAMA_UBATCH must be at least 1, got {value}")
        return value
    raw = os.environ.get("SEMIF_LLAMA_UBATCH_PER_SEQ", "512").strip()
    try:
        per_sequence = int(raw)
    except ValueError:
        raise ValueError(f"SEMIF_LLAMA_UBATCH_PER_SEQ {raw!r} is not an integer") from None
    if per_sequence < 1:
        raise ValueError(f"SEMIF_LLAMA_UBATCH_PER_SEQ must be at least 1, got {per_sequence}")
    return max(512, sequences * per_sequence)


def _default_threads() -> int:
    """Default thread count: mild oversubscription of the hardware threads.

    Deliberately more threads than the CPU has. When the checkpoint does not fit in
    RAM the decode stalls on expert-weight page faults, and a thread waiting on a
    fault is not runnable — so a decode served from disk wants *more* runnable
    threads than hardware threads, to keep the device queue fed. Measured over one
    fixed pass (908 token-forwards, ~41 GB read, 6-core/12-thread host):

        6 threads 54.6 s | 8 threads 50.4 s | 12 threads 47.9 s
        16 threads 38.4 s | 20 threads 38.4 s

    The knee is 4/3 of the hardware threads, which is what this returns. Note the
    previous default (`os.cpu_count()`) was measurably worse, and the project's own
    runner pinned 6 — half this host — for no recorded reason.
    """
    return max(4, (os.cpu_count() or 4) * 4 // 3)


def _cpu_model_params(library):
    """Initialize llama.cpp once and return CPU-only, disk-streamed model parameters.

    Expert weights stay file-backed and demand-paged. Two fields control that, and
    neither is ``use_mmap`` / ``use_mlock``: those were removed from
    ``llama_model_params`` in this llama.cpp, and because a ctypes ``Structure``
    accepts arbitrary Python attributes, assigning them succeeds while llama.cpp
    never reads them -- a silent no-op. The real controls are:

    ``load_mode``
        ``mmap`` keeps the GGUF mapped, so a routed expert is read from disk on
        first touch and its page stays evictable. ``mlock`` / ``mmap_mlock`` would
        pin the weights in RAM, and a checkpoint larger than memory then fails to
        load at all rather than merely running slow.

    ``use_extra_bufts``
        Enables the CPU backend's extra buffer types, which include the *repack*
        buffer. Repack rewrites quantized expert tensors into an interleaved
        SIMD-friendly layout (``q4_K`` -> ``q4_K_8x8``) **at load time**,
        materialising them into anonymous memory and defeating mmap streaming
        entirely -- the failure is an OOM kill during load, not slow inference.
        It is therefore off by default here, since streaming is this backend's
        point. Turn it back on (``SEMIF_LLAMA_EXTRA_BUFT=1``) when the checkpoint
        fits in RAM and prompt-processing speed matters more than residency.

    Both are environment-overridable so a run can be pinned without editing the
    scorer.
    """
    global _BACKEND_INITIALIZED
    if not _BACKEND_INITIALIZED:
        library.llama_backend_init()
        _BACKEND_INITIALIZED = True
    params = library.llama_model_default_params()
    params.n_gpu_layers = 0
    params.load_mode = _load_mode()
    params.use_extra_bufts = _env_flag("SEMIF_LLAMA_EXTRA_BUFT", False)
    return params


def _render(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _gguf_tokenize(library, vocab, text: str) -> list[int]:
    data = text.encode("utf-8")
    needed = library.llama_tokenize(vocab, data, len(data), None, 0, False, True)
    if needed < 0:
        needed = -needed
    tokens = (library.llama_token * needed)()
    written = library.llama_tokenize(vocab, data, len(data), tokens, needed, False, True)
    if written < 0:
        raise RuntimeError("The GGUF tokenizer rejected the prompt text")
    return list(tokens[:written])


def _gguf_piece(library, vocab, token: int) -> bytes:
    buffer = ctypes.create_string_buffer(64)
    written = library.llama_token_to_piece(vocab, token, buffer, len(buffer), 0, True)
    if written < 0:
        raise RuntimeError("The GGUF tokenizer cannot render a token")
    return buffer.raw[:written]


def _logsumexp(values: numpy.ndarray) -> float:
    peak = float(values.max())
    return peak + float(numpy.log(numpy.exp(values - peak).sum()))


class _Engine:
    """One llama.cpp context bound to a loaded GGUF model."""

    def __init__(self, library, model, context_tokens: int, threads: int, sequences: int = 1,
                 ubatch: int | None = None):
        params = library.llama_context_default_params()
        params.n_ctx = context_tokens
        if ubatch is None:
            ubatch = _ubatch(sequences)
        # Branches are scored as parallel sequences carrying the whole prompt, so
        # the context needs room for len(rows) live sequences rather than 1: with
        # n_seq_max = 1 the only way to score N criteria is N sequential passes,
        # each re-reading every expert from disk, which is what made scoring cost
        # linear in the criteria count.
        params.n_seq_max = sequences
        # llama.cpp requires n_outputs_max >= n_seq_max * n_outputs_max_per_seq.
        params.n_outputs_max = sequences
        # The physical batch is the unit that actually decides I/O. A decode of
        # more tokens than n_ubatch is split into several ubatches, and each
        # ubatch walks all layers and re-reads their experts — so leaving this at
        # llama.cpp's default of 512 turned a 908-token branch group into two
        # full model sweeps. Sized here to hold one whole group in one ubatch.
        params.n_batch = params.n_ubatch = ubatch
        params.n_threads = threads
        params.n_threads_batch = threads
        self.lib = library
        self.sequences = sequences
        self.ubatch = int(params.n_ubatch)
        self.model = model
        self.context = library.llama_init_from_model(model, params)
        if not self.context:
            raise RuntimeError("llama.cpp failed to create the scoring context")
        self.memory = library.llama_get_memory(self.context)
        if not self.memory:
            library.llama_free(self.context)
            self.context = None
            raise RuntimeError("llama.cpp returned no context memory")
        self.context_tokens = int(library.llama_n_ctx(self.context))
        self.vocab_size = library.llama_n_vocab(library.llama_model_get_vocab(model))

    def close(self) -> None:
        if self.context:
            self.lib.llama_free(self.context)
            self.context = None
            self.memory = None

    def _decode(self, tokens: list[int], start: int, sequence: int, want_logits: bool):
        if not tokens:
            raise ValueError("Refusing to decode an empty token list")
        total = len(tokens)
        for offset in range(0, total, DECODE_CHUNK):
            chunk = tokens[offset : offset + DECODE_CHUNK]
            batch = self.lib.llama_batch_init(len(chunk), 0, 1)
            try:
                for index in range(len(chunk)):
                    batch.token[index] = chunk[index]
                    batch.pos[index] = start + offset + index
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = sequence
                    batch.logits[index] = int(want_logits and offset + index == total - 1)
                batch.n_tokens = len(chunk)
                if self.lib.llama_decode(self.context, batch):
                    raise RuntimeError("llama_decode failed; raise --max-tokens if prompts grew")
            finally:
                self.lib.llama_batch_free(batch)
        if not want_logits:
            return None
        pointer = self.lib.llama_get_logits_ith(self.context, -1)
        if not pointer:
            raise RuntimeError("llama.cpp returned no logits for the flagged position")
        return numpy.ctypeslib.as_array(
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)), shape=(self.vocab_size,)
        ).copy()

    def clear(self) -> None:
        self.lib.llama_memory_clear(self.memory, False)

    def prefill(self, prefix: list[int]) -> None:
        self._decode(prefix, 0, 0, False)

    def save_state(self):
        """Snapshot sequence 0 for repeated branch restores."""
        size = self.lib.llama_state_seq_get_size(self.context, 0)
        if size <= 0:
            raise RuntimeError("llama.cpp returned an empty prefix state")
        buffer = (ctypes.c_ubyte * size)()
        if self.lib.llama_state_seq_get_data(self.context, buffer, size, 0) != size:
            raise RuntimeError("llama.cpp wrote an incomplete prefix state")
        return buffer, size

    def restore_state(self, state) -> None:
        buffer, size = state
        if not self.lib.llama_memory_seq_rm(self.memory, 0, -1, -1):
            raise RuntimeError("llama.cpp could not drop the previous scored branch")
        if self.lib.llama_state_seq_set_data(self.context, buffer, size, 0) == 0:
            raise RuntimeError("llama.cpp could not restore the saved prefix state")

    def branch_logits(self, prefix_length: int, suffix: list[int]) -> numpy.ndarray:
        return self._decode(suffix, prefix_length, 0, True)

    def branch_logits_batched(self, prefix: list[int], suffixes: list[list[int]]):
        """Score every branch in one batched forward pass, one logits row each.

        Each branch is a sequence carrying the *whole* prompt — the shared prefix
        plus its own suffix — and all of them ride in a single ``llama_decode``.
        That is what makes scoring cost independent of the criteria count on the
        SSD path: the routed experts for the whole set are read once per forward
        pass instead of once per criterion.

        The cheaper-looking alternative — prefill the prefix once and copy its KV
        into each branch with ``llama_memory_seq_cp`` — does not work here. This
        checkpoint is a *hybrid*: ``qwen35moe`` interleaves recurrent
        (linear-attention) layers with attention layers, and llama.cpp's
        ``llama_memory_hybrid::seq_cp`` aborts on the recurrent half (measured:
        SIGABRT in ``llama_kv_cache::seq_cp``). Recomputing the prefix per branch
        is compute, and compute is not the term that dominates here — I/O is.

        Positions restart at 0 for every branch because each sequence owns its own
        KV; the memory is cleared by the caller before this runs.
        """
        if not suffixes:
            raise ValueError("Refusing to score an empty branch set")
        if len(suffixes) > self.sequences:
            raise ValueError(
                f"{len(suffixes)} branches exceed the context's n_seq_max={self.sequences}; "
                "raise SEMIF_LLAMA_SEQ_MAX or score in smaller groups"
            )
        if not prefix:
            raise ValueError("Refusing to score over an empty prefix")
        if any(not suffix for suffix in suffixes):
            raise ValueError("Refusing to score an empty branch")
        # Flat work list: (token, position, sequence, is_last_for_that_sequence).
        pending = []
        for sequence, suffix in enumerate(suffixes):
            tokens = list(prefix) + list(suffix)
            last = len(tokens) - 1
            pending.extend(
                (token, position, sequence, position == last)
                for position, token in enumerate(tokens)
            )
        logits: list = [None] * len(suffixes)
        # Chunk at the physical batch size, not DECODE_CHUNK: handing llama.cpp
        # more tokens than one ubatch only makes it split internally, which is
        # the extra model sweep this is here to avoid.
        for offset in range(0, len(pending), self.ubatch):
            chunk = pending[offset : offset + self.ubatch]
            batch = self.lib.llama_batch_init(len(chunk), 0, 1)
            marked = []
            try:
                for index, (token, position, sequence, is_last) in enumerate(chunk):
                    batch.token[index] = token
                    batch.pos[index] = position
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index][0] = sequence
                    batch.logits[index] = int(is_last)
                    if is_last:
                        marked.append((index, sequence))
                batch.n_tokens = len(chunk)
                if self.lib.llama_decode(self.context, batch):
                    raise RuntimeError("llama_decode failed; raise --max-tokens if prompts grew")
            finally:
                self.lib.llama_batch_free(batch)
            for index, sequence in marked:
                pointer = self.lib.llama_get_logits_ith(self.context, index)
                if not pointer:
                    raise RuntimeError("llama.cpp returned no logits for a scored branch")
                logits[sequence] = numpy.ctypeslib.as_array(
                    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float)), shape=(self.vocab_size,)
                ).copy()
        if any(row is None for row in logits):
            raise RuntimeError("llama.cpp returned no logits for a scored branch")
        return logits

    def full_logits(self, tokens: list[int]) -> numpy.ndarray:
        self.clear()
        return self._decode(tokens, 0, 0, True)


def _free_native(engine: _Engine, library, model) -> None:
    engine.close()
    library.llama_model_free(model)


class _Backend:
    """Verified scoring adapter around one llama.cpp engine."""

    def __init__(self, engine: _Engine, model, vocab, tokenizer):
        self.engine = engine
        self.vocab = vocab
        self.tokenizer = tokenizer
        self._finalizer = weakref.finalize(self, _free_native, engine, engine.lib, model)

    def close(self) -> None:
        self._finalizer()

    def encode_verified(self, row: dict, max_tokens: int):
        ids, slots, prompt_hash = encode_prompt(self.tokenizer, row, max_tokens)
        if _gguf_tokenize(self.engine.lib, self.vocab, _render(self.tokenizer, row)) != ids:
            raise ValueError(f"Row {row['id']}: GGUF tokenization disagrees with the reference tokenizer")
        return ids, slots, prompt_hash


def _verify_vocabulary(tokenizer, library, vocab) -> None:
    """Fail early when the GGUF vocabulary is not the tokenizer's own."""
    row = {
        "id": "vocabulary-probe",
        "state": "probe evidence",
        "question": "probe criterion?",
        "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}],
    }
    prompt = _render(tokenizer, row)
    reference = tokenizer.encode(prompt, add_special_tokens=False)
    if _gguf_tokenize(library, vocab, prompt) != reference:
        raise RuntimeError("The GGUF vocabulary disagrees with the reference tokenizer")
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or _gguf_piece(library, vocab, encoded[0]) != letter.encode():
            raise RuntimeError(f"Answer slot {letter!r} is not a shared single token")


def _gguf_digest(gguf: Path) -> tuple[str, str]:
    """Return ``(sha256, source)`` for the checkpoint, reusing a sidecar when valid.

    The digest is an integrity record, and computing it reads the *whole* file —
    20.9 GB on this checkpoint. That hurts twice over: llama.cpp's own loader then
    reads the file again (measured: it reads all of it even under ``mmap``), so a
    cold run paid two full sweeps before scoring a single token.

    So the digest is cached beside the checkpoint as ``<name>.sha256.json``, keyed
    on size + ``mtime_ns``. Any change to either invalidates it and the file is
    hashed again. ``SEMIF_GGUF_DIGEST=compute`` forces a fresh hash;
    ``SEMIF_GGUF_DIGEST=skip`` records no digest at all rather than a stale one.
    Whichever path is taken is named in the result metadata, so a row always says
    how much its own integrity record can be trusted.
    """
    mode = os.environ.get("SEMIF_GGUF_DIGEST", "cache").strip().lower()
    if mode not in {"cache", "compute", "skip"}:
        raise ValueError(
            f"unknown SEMIF_GGUF_DIGEST {mode!r}; expected cache, compute or skip"
        )
    if mode == "skip":
        return "", "skipped"
    stat = gguf.stat()
    sidecar = gguf.with_name(gguf.name + ".sha256.json")
    if mode == "cache":
        try:
            record = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            record = None
        if (isinstance(record, dict) and record.get("sha256")
                and record.get("bytes") == stat.st_size
                and record.get("mtime_ns") == stat.st_mtime_ns):
            return str(record["sha256"]), "cache"
    checksum = hashlib.sha256()
    with gguf.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    digest = checksum.hexdigest()
    try:
        sidecar.write_text(json.dumps(
            {"sha256": digest, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        ))
    except OSError:
        pass  # A read-only model directory is fine; the digest is still returned.
    return digest, "computed"


def load_model(source: str, revision: str, gguf, *, threads: int | None = None,
               context_tokens: int = 4096):
    """Load one pinned reference tokenizer plus a local GGUF checkpoint for CPU scoring."""
    local = Path(source).is_dir()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote sources require a pinned 40-character revision; local sources require a revision label")
    if local and not revision:
        raise ValueError("Local sources require an explicit revision label")
    gguf = Path(gguf)
    if not gguf.is_file():
        raise ValueError(f"GGUF checkpoint not found: {gguf}")
    if not (isinstance(context_tokens, int) and context_tokens > 0):
        raise ValueError("context_tokens must be a positive integer")
    if threads is None:
        threads = _default_threads()
    if not (isinstance(threads, int) and threads >= 1):
        raise ValueError("threads must be a positive integer")
    import transformers
    try:
        import llama_cpp
    except ImportError as error:
        raise RuntimeError("Install the llama.cpp extra: pip install -e '.[test,llamacpp]'") from error

    offline = bool(os.environ.get("HF_HUB_OFFLINE"))
    common = {"revision": None if local else revision,
              "local_files_only": local or offline, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    # Finish file I/O before allocating native resources so read failures cannot leak them.
    digest, digest_source = _gguf_digest(gguf)
    gguf_record = {"file": gguf.name, "bytes": gguf.stat().st_size,
                   "sha256": digest, "digest_source": digest_source}
    model_params = _cpu_model_params(llama_cpp)
    model = llama_cpp.llama_model_load_from_file(str(gguf).encode("utf-8"), model_params)
    if not model:
        raise RuntimeError(f"llama.cpp failed to load the GGUF checkpoint: {gguf}")
    window = context_tokens + 64
    sequences = _seq_max()
    try:
        engine = _Engine(llama_cpp, model, window, threads, sequences)
        vocab = llama_cpp.llama_model_get_vocab(model)
        _verify_vocabulary(tokenizer, llama_cpp, vocab)
    except Exception:
        if "engine" in locals():
            engine.close()
        llama_cpp.llama_model_free(model)
        raise
    metadata = {
        "source": source,
        "revision": revision,
        "backend": "llamacpp",
        "dtype": "gguf-quantized",
        "gguf": gguf_record,
        "vocab_size": engine.vocab_size,
        "threads": threads,
        "n_gpu_layers": 0,
        "load_mode": int(model_params.load_mode),
        "load_mode_name": _LOAD_MODE_NAMES.get(int(model_params.load_mode), "unknown"),
        "use_extra_bufts": bool(model_params.use_extra_bufts),
        "max_prompt_tokens": context_tokens,
        "context_tokens": engine.context_tokens,
        "n_seq_max": engine.sequences,
        "n_ubatch": engine.ubatch,
        "decode_chunk": DECODE_CHUNK,
        "llama_cpp_python_version": llama_cpp.__version__,
        "transformers_version": transformers.__version__,
    }
    return _Backend(engine, model, vocab, tokenizer), tokenizer, metadata


def _result(row: dict, encoded, selected: list[float], vocabulary, metadata: dict, config: str, readout: str) -> dict:
    ids, slots, prompt_hash = encoded
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "answer_token_ids": slots,
        "input_tokens": len(ids),
        "allowed_token_mass": float(numpy.exp(_logsumexp(numpy.asarray(selected)) - _logsumexp(vocabulary))),
        "full_vocab_argmax_id": int(vocabulary.argmax()),
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": {**metadata, "serving_config": config},
        "readout": readout,
        "probability_status": "conditional option score over quantized weights; uncalibrated as decision confidence",
    }


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    started = time.perf_counter()
    encoded = model.encode_verified(row, max_tokens)
    mark = time.perf_counter()
    vocabulary = model.engine.full_logits(encoded[0])
    selected = vocabulary[encoded[1]].tolist()
    result = _result(
        row, encoded, selected, vocabulary, metadata, "llamacpp-direct-v1",
        "quantized last-position logits restricted to declared answer slots; no generated tokens",
    )
    result.update(forward_seconds=time.perf_counter() - mark, total_seconds=time.perf_counter() - started)
    return result


class SerialPrefixScorer:
    """Cache the current state once, then score restored-state branch suffixes."""

    def __init__(self, model, tokenizer, metadata: dict, max_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = {**metadata, "serving_config": "llamacpp-state-restore-v1"}
        self.max_tokens = max_tokens
        self.prefix = None
        self.state_data = None

    def score(self, row: dict) -> dict:
        started = time.perf_counter()
        encoded = self.model.encode_verified(row, self.max_tokens)
        ids, slots, _ = encoded
        prefix = _state_prefix(self.tokenizer, row["state"])
        hit = self.state_data is not None and prefix == self.prefix
        if not prefix or ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
            raise ValueError("State prefix does not match the full prompt")
        prefill_seconds = 0.0
        if not hit:
            mark = time.perf_counter()
            self.model.engine.clear()
            self.model.engine.prefill(prefix)
            prefill_seconds = time.perf_counter() - mark
            self.prefix = prefix
            self.state_data = self.model.engine.save_state()
        mark = time.perf_counter()
        self.model.engine.restore_state(self.state_data)
        copy_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        vocabulary = self.model.engine.branch_logits(len(prefix), ids[len(prefix) :])
        suffix_seconds = time.perf_counter() - mark
        selected = vocabulary[slots].tolist()
        result = _result(
            row, encoded, selected, vocabulary, self.metadata,
            "llamacpp-state-restore-v1", "quantized branch last-position logits over a restored prefix state",
        )
        result.update(
            cache_hit=hit,
            prefix_tokens=len(prefix),
            prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            branch_state_bytes=self.state_data[1],
            prefill_seconds=prefill_seconds,
            copy_seconds=copy_seconds,
            suffix_forward_seconds=suffix_seconds,
            forward_seconds=prefill_seconds + suffix_seconds,
            total_seconds=time.perf_counter() - started,
        )
        return result


def score_shared(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096):
    """Prefill one exact state once, then score every criterion from restored branches."""
    if not rows or any(row["state"] != rows[0]["state"] for row in rows[1:]):
        raise ValueError("Shared scoring requires one nonempty exact state")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    started = time.perf_counter()
    encoded = [model.encode_verified(row, max_tokens) for row in rows]
    prefix = _state_prefix(tokenizer, rows[0]["state"])
    if not prefix or any(ids[: len(prefix)] != prefix or len(ids) <= len(prefix) for ids, _, _ in encoded):
        raise ValueError("The fixed state prefix does not match every full prompt")
    encode_seconds = time.perf_counter() - started
    # One forward pass per group of criteria rather than one per criterion: every
    # branch carries the whole prompt as its own sequence inside a single
    # llama_decode, so the routed experts are read once for the whole group
    # instead of once per criterion. There is no separate prefix prefill and no
    # per-branch state restore — both existed only to work around a
    # single-sequence context, which is exactly what made the cost linear in the
    # row count.
    width = max(1, model.engine.sequences)
    forward_seconds = 0.0
    results = []
    for start in range(0, len(rows), width):
        group = list(zip(rows, encoded))[start : start + width]
        # Release the previous group's cells before its sequence ids are reused. Every
        # branch carries the whole prompt as its own sequence, so positions restart at 0
        # and a sequence still holding the previous group's tokens fails the decode
        # outright — measured: 16 criteria against n_seq_max=8 raised "llama_decode
        # failed" from the second group onward. Only reachable beyond n_seq_max criteria.
        model.engine.clear()
        mark = time.perf_counter()
        vocabularies = model.engine.branch_logits_batched(
            prefix, [ids[len(prefix):] for _, (ids, _, _) in group]
        )
        forward_seconds += time.perf_counter() - mark
        for (row, row_encoded), vocabulary in zip(group, vocabularies):
            _, slots, _ = row_encoded
            results.append(_result(
                row, row_encoded, vocabulary[slots].tolist(), vocabulary, metadata,
                "llamacpp-batched-prefill-shared-v2",
                "quantized batched full-prompt last-position logits over one shared state",
            ))
    suffix_total = sum(len(ids) - len(prefix) for ids, _, _ in encoded)
    timing = {
        "total_seconds": time.perf_counter() - started,
        "encode_seconds": encode_seconds,
        "prefix_tokens": len(prefix),
        "prefill_seconds": 0.0,
        "replicate_seconds": 0.0,
        "suffix_forward_seconds": forward_seconds,
        "batched_forward_seconds": forward_seconds,
        "batch_size": len(rows),
        "batched_passes": (len(rows) + width - 1) // width,
        "batched_tokens": suffix_total + len(prefix) * len(rows),
        "true_suffix_tokens": suffix_total,
        "padded_suffix_tokens": suffix_total,
    }
    return results, timing
