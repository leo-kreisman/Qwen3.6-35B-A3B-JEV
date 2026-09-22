"""Batched branch scoring: the shared-state path that avoids one pass per criterion.

These cover the two mechanisms that made scoring cost linear in the criteria
count — a single-sequence context and a physical batch smaller than the branch
group — plus the digest cache that stopped a cold run reading the checkpoint
twice before scoring a token.
"""
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from semif_phase1 import llamacpp_backend


class _FakeBatch:
    """Minimal stand-in for a llama_batch: indexable parallel arrays."""

    def __init__(self, size):
        self.token = [0] * size
        self.pos = [0] * size
        self.n_seq_id = [0] * size
        self.seq_id = [[0] for _ in range(size)]
        self.logits = [0] * size
        self.n_tokens = 0


class _FakeLibrary:
    """Records every batch handed to llama_decode and returns per-index logits."""

    def __init__(self, vocab_size=4):
        self.batches = []
        self.vocab_size = vocab_size
        self._buffers = []

    def llama_batch_init(self, n_tokens, embd, n_seq_max):
        return _FakeBatch(n_tokens)

    def llama_batch_free(self, batch):
        pass

    def llama_decode(self, context, batch):
        self.batches.append(batch)
        return 0

    def llama_get_logits_ith(self, context, index):
        # A distinct leading value per batch index, so the caller's mapping from
        # batch slot to sequence can be checked rather than assumed.
        values = [float(index)] + [0.0] * (self.vocab_size - 1)
        buffer = (ctypes.c_float * self.vocab_size)(*values)
        self._buffers.append(buffer)
        return ctypes.cast(buffer, ctypes.POINTER(ctypes.c_float))


def _engine(sequences=4, ubatch=4096, vocab_size=4):
    engine = llamacpp_backend._Engine.__new__(llamacpp_backend._Engine)
    engine.lib = _FakeLibrary(vocab_size)
    engine.sequences = sequences
    engine.ubatch = ubatch
    engine.vocab_size = vocab_size
    engine.memory = object()
    engine.context = object()
    return engine


def test_batched_branches_ride_one_decode_with_per_sequence_positions():
    engine = _engine()
    logits = engine.branch_logits_batched([1, 2], [[9], [8, 7]])
    (batch,) = engine.lib.batches
    assert batch.n_tokens == 7
    # Every branch restarts at position 0 because each owns its own KV.
    assert batch.pos[:7] == [0, 1, 2, 0, 1, 2, 3]
    assert [batch.seq_id[i][0] for i in range(7)] == [0, 0, 0, 1, 1, 1, 1]
    # Exactly the last token of each branch is flagged for logits.
    assert batch.logits[:7] == [0, 0, 1, 0, 0, 0, 1]
    assert [row[0] for row in logits] == [2.0, 6.0]


def test_batched_branches_still_flag_each_end_when_split_across_ubatches():
    engine = _engine(ubatch=3)
    logits = engine.branch_logits_batched([1, 2], [[9], [8, 7]])
    # Three chunks of three, two and two; the second branch ends in the last one.
    assert [batch.n_tokens for batch in engine.lib.batches] == [3, 3, 1]
    assert [batch.logits[: batch.n_tokens] for batch in engine.lib.batches] == [
        [0, 0, 1], [0, 0, 0], [1],
    ]
    # llama_get_logits_ith is indexed within its own decode, so the second
    # branch — which ends at the start of the third chunk — comes back at 0.
    assert [row[0] for row in logits] == [2.0, 0.0]
    assert len(logits) == 2


def test_batched_branches_sizes_a_whole_group_into_one_physical_batch():
    assert llamacpp_backend._ubatch(8) == 4096
    assert llamacpp_backend._ubatch(4) == 2048
    # Never below llama.cpp's own default, even for a single branch.
    assert llamacpp_backend._ubatch(1) == 512


@pytest.mark.parametrize("cpus,expected", [
    (12, 16),   # the host this was measured on; knee of the sweep
    (8, 10),
    (6, 8),
    (4, 5),
    (2, 4),     # never below the floor
    (None, 5),  # falls back to a 4-core assumption, then the same 4/3 rule
])
def test_default_threads_oversubscribes_hardware_threads(monkeypatch, cpus, expected):
    # A decode stalled on expert page faults wants more runnable threads than the CPU
    # has, so the default is 4/3 of nproc rather than nproc.
    monkeypatch.setattr(llamacpp_backend.os, "cpu_count", lambda: cpus)
    assert llamacpp_backend._default_threads() == expected


def test_score_shared_clears_between_groups_so_sequence_ids_can_be_reused(monkeypatch):
    """More criteria than n_seq_max run as several groups over the same sequence ids.

    Each branch carries the whole prompt, so positions restart at 0 for every group.
    If a group does not release the previous group's cells first, reusing sequence 0
    for a shorter prompt fails the decode outright — measured on the real checkpoint:
    16 criteria against n_seq_max=8 raised "llama_decode failed" from group 2 onward.
    """

    class _FakeEngine:
        def __init__(self):
            self.sequences = 2
            self.clears = 0
            self.groups = []

        def clear(self):
            self.clears += 1

        def branch_logits_batched(self, prefix, suffixes):
            self.groups.append(len(suffixes))
            return [numpy.asarray([0.0, 1.0, 2.0, 3.0]) + index
                    for index in range(len(suffixes))]

    class _FakeModel:
        def __init__(self):
            self.engine = _FakeEngine()

        def encode_verified(self, row, max_tokens):
            return [1, 2, 9], [0, 1], "hash"

    monkeypatch.setattr(llamacpp_backend, "_state_prefix", lambda tokenizer, state: [1, 2])
    model = _FakeModel()
    rows = [
        {"id": f"r{index}", "state": {"a": 1}, "question": "Which option?",
         "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]}
        for index in range(5)
    ]
    results, timing = llamacpp_backend.score_shared(model, object(), rows, {}, 4096)
    assert [row["id"] for row in results] == ["r0", "r1", "r2", "r3", "r4"]
    # 5 rows over a 2-sequence context is 3 groups, and every one clears first.
    assert model.engine.groups == [2, 2, 1]
    assert model.engine.clears == 3
    assert timing["batched_passes"] == 3
    assert all(sum(row["probabilities"]) == pytest.approx(1.0) for row in results)


@pytest.mark.parametrize("env,expected", [
    ({}, 8),
    ({"SEMIF_LLAMA_SEQ_MAX": "3"}, 3),
])
def test_seq_max_resolution(monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert llamacpp_backend._seq_max() == expected


@pytest.mark.parametrize("name,value,message", [
    ("SEMIF_LLAMA_SEQ_MAX", "many", "not an integer"),
    ("SEMIF_LLAMA_SEQ_MAX", "0", "at least 1"),
    ("SEMIF_LLAMA_UBATCH", "wide", "not an integer"),
    ("SEMIF_LLAMA_UBATCH", "0", "at least 1"),
    ("SEMIF_LLAMA_UBATCH_PER_SEQ", "0", "at least 1"),
])
def test_batch_controls_reject_bad_values(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    target = llamacpp_backend._seq_max if name.endswith("SEQ_MAX") else (
        lambda: llamacpp_backend._ubatch(2)
    )
    with pytest.raises(ValueError, match=message):
        target()


def test_batched_branches_refuse_more_branches_than_the_context_holds():
    engine = _engine(sequences=2)
    with pytest.raises(ValueError, match="n_seq_max=2"):
        engine.branch_logits_batched([1], [[9], [8], [7]])
    assert engine.lib.batches == []


@pytest.mark.parametrize("prefix,suffixes,message", [
    ([1], [], "empty branch set"),
    ([], [[9]], "empty prefix"),
    ([1], [[]], "empty branch"),
])
def test_batched_branches_refuse_degenerate_input(prefix, suffixes, message):
    engine = _engine()
    with pytest.raises(ValueError, match=message):
        engine.branch_logits_batched(prefix, suffixes)
    assert engine.lib.batches == []


def test_engine_sizes_context_for_the_branch_group(monkeypatch):
    monkeypatch.setenv("SEMIF_LLAMA_SEQ_MAX", "4")
    captured = {}

    library = SimpleNamespace(
        llama_context_default_params=lambda: SimpleNamespace(),
        llama_init_from_model=lambda model, params: captured.update(vars(params)) or object(),
        llama_get_memory=lambda value: object(),
        llama_n_ctx=lambda value: 4096,
        llama_model_get_vocab=lambda model: object(),
        llama_n_vocab=lambda vocab: 8,
    )
    engine = llamacpp_backend._Engine(library, object(), 4096, 2, 4)
    assert captured["n_seq_max"] == 4
    assert captured["n_outputs_max"] == 4
    # A group split across ubatches pays a full model sweep each, so the physical
    # batch must be able to hold one whole group.
    assert captured["n_ubatch"] == 2048
    assert captured["n_batch"] == 2048
    assert engine.ubatch == 2048


def test_digest_cache_reuses_an_unchanged_checkpoint(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"weights" * 64)
    first, source = llamacpp_backend._gguf_digest(gguf)
    assert source == "computed"
    again, source = llamacpp_backend._gguf_digest(gguf)
    assert (again, source) == (first, "cache")
    # A change that alters neither size nor mtime is the one case the sidecar
    # cannot see; anything else must rehash.
    gguf.write_bytes(b"weights" * 128)
    third, source = llamacpp_backend._gguf_digest(gguf)
    assert source == "computed"
    assert third != first


def test_digest_can_be_forced_or_skipped(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"weights")
    llamacpp_backend._gguf_digest(gguf)
    monkeypatch.setenv("SEMIF_GGUF_DIGEST", "compute")
    assert llamacpp_backend._gguf_digest(gguf)[1] == "computed"
    monkeypatch.setenv("SEMIF_GGUF_DIGEST", "skip")
    assert llamacpp_backend._gguf_digest(gguf) == ("", "skipped")
    monkeypatch.setenv("SEMIF_GGUF_DIGEST", "sometimes")
    with pytest.raises(ValueError, match="SEMIF_GGUF_DIGEST"):
        llamacpp_backend._gguf_digest(gguf)


def test_digest_survives_a_read_only_model_directory(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"weights")
    directory = tmp_path / "readonly"
    directory.mkdir()
    sealed = directory / "model.gguf"
    sealed.write_bytes(b"weights")
    directory.chmod(0o500)
    try:
        digest, source = llamacpp_backend._gguf_digest(sealed)
        assert source == "computed"
        assert len(digest) == 64
    finally:
        directory.chmod(0o700)
