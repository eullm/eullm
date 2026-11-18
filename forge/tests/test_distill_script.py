"""Tests for the Leonardo distillation script's bookkeeping.

`forge/scripts/distill.py` is a hand-written training loop, so the parts that
are not the loop itself carry no framework behind them and get no framework's
testing either. Two of them can destroy a run:

  * `prune_checkpoints` deletes directories. A run's only way back from a
    walltime kill is the checkpoint it last wrote, so an off-by-one here does
    not produce a wrong number, it produces no way to resume.
  * `_checkpoint_step` decides which checkpoint is newest. Sorted as strings,
    `checkpoint-9000` comes after `checkpoint-18000`, and a resume would
    silently redo nine thousand steps — three days of Booster time.

Both are pure filesystem logic, so they are tested against real directories.
`evaluate` is tested against stand-in models, since the real pair is 34 B
parameters and needs four GPUs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "distill.py"


def _load():
    spec = importlib.util.spec_from_file_location("distill_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves `cls.__module__` through
    # sys.modules, and without this the config class fails to build.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


distill = _load()


def make_checkpoints(root: Path, steps) -> None:
    for step in steps:
        d = root / f"checkpoint-{step}"
        d.mkdir(parents=True)
        write_state(d / "training_state.pt")


def write_state(path: Path) -> None:
    """A training_state.pt as torch.save leaves it: a zip archive."""
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", b"x")


# --- the CLI gives every flag the type the config declares --------------------

def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["distill.py", *argv])
    return distill._parse_args()


def test_a_numeric_flag_arrives_as_a_number(monkeypatch):
    """"3" stayed a string, and range() said so deep in the trainer.

    The module has `from __future__ import annotations`, so a dataclass field's
    .type is the string "int" under PEP 563 and the identity tests in
    _parse_args never fired: every flag fell through to the str branch.
    """
    cfg = _parse(monkeypatch, "--num-train-epochs", "3", "--learning-rate", "1e-4")
    assert cfg.num_train_epochs == 3 and isinstance(cfg.num_train_epochs, int)
    assert cfg.learning_rate == pytest.approx(1e-4) and isinstance(cfg.learning_rate, float)
    assert list(range(cfg.num_train_epochs)) == [0, 1, 2]


def test_a_bool_flag_can_be_turned_off(monkeypatch):
    """--bf16 false used to leave bf16 true, with nothing said.

    A bool field never got BooleanOptionalAction, so --bf16 took a value:
    the string "false", which bool() reads as true.
    """
    assert _parse(monkeypatch, "--no-bf16").bf16 is False
    assert _parse(monkeypatch, "--bf16").bf16 is True


def test_a_flag_the_config_does_not_type_still_arrives_as_a_string(monkeypatch):
    """The list fields are split by the trainer, not by argparse."""
    cfg = _parse(monkeypatch, "--student-lora-target-modules", "q_proj,v_proj")
    assert cfg.student_lora_target_modules == "q_proj,v_proj"


# ── which checkpoint is the newest ───────────────────────────────────────

def test_step_is_read_as_a_number_not_a_string(tmp_path):
    """The bug this guards: '9000' > '18000' lexicographically."""
    make_checkpoints(tmp_path, [9000, 18000])
    assert distill.latest_checkpoint(tmp_path).name == "checkpoint-18000"


def test_unnumbered_directories_never_win(tmp_path):
    make_checkpoints(tmp_path, [500])
    (tmp_path / "checkpoint-final").mkdir()
    assert distill.latest_checkpoint(tmp_path).name == "checkpoint-500"


def test_a_checkpoint_a_kill_caught_mid_write_is_skipped(tmp_path, capsys):
    """The adapter is written first, the optimizer state second.

    A walltime kill between the two leaves a checkpoint-N/ with the weights
    and no training_state.pt, and load_checkpoint reads that file. Returning
    it costs the rest of the chain: the link dies, the next link picks the same
    directory, and the complete checkpoints that would have loaded are never
    tried. The comment above prune_checkpoints already said the run's way back
    is the one before it.
    """
    make_checkpoints(tmp_path, [2, 4, 6])
    cut_short = tmp_path / "checkpoint-400"
    cut_short.mkdir()
    (cut_short / "adapter_model.safetensors").write_bytes(b"pesi")

    assert distill.latest_checkpoint(tmp_path).name == "checkpoint-6"
    assert "checkpoint-400" in capsys.readouterr().err


def test_a_state_file_cut_short_by_a_kill_is_skipped(tmp_path, capsys):
    """norms-v2, 17-10-2026: a link was killed during torch.save, and the
    truncated training_state.pt of checkpoint-1100 failed every later link
    of the 8B at once ("failed finding central directory")."""
    make_checkpoints(tmp_path, [1000, 1100])
    state = tmp_path / "checkpoint-1100" / "training_state.pt"
    whole = state.read_bytes()
    state.write_bytes(whole[:len(whole) // 2])

    assert distill.latest_checkpoint(tmp_path).name == "checkpoint-1000"
    assert "truncated" in capsys.readouterr().err


def test_only_half_written_checkpoints_resumes_nothing_at_all(tmp_path):
    make_checkpoints(tmp_path, [])
    for step in (7, 8):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert distill.latest_checkpoint(tmp_path) is None


def test_no_checkpoints_is_none_not_an_error(tmp_path):
    assert distill.latest_checkpoint(tmp_path) is None
    assert distill.latest_checkpoint(tmp_path / "absent") is None


# ── pruning ──────────────────────────────────────────────────────────────

def test_keeps_the_newest_n_by_step(tmp_path):
    make_checkpoints(tmp_path, [300, 600, 900, 1200, 1500])
    distill.prune_checkpoints(tmp_path, keep=3)
    left = sorted(p.name for p in tmp_path.glob("checkpoint-*"))
    assert left == ["checkpoint-1200", "checkpoint-1500", "checkpoint-900"]


def test_pruning_ranks_by_number_too(tmp_path):
    """Keeping the three 'largest' as strings would drop checkpoint-18000."""
    make_checkpoints(tmp_path, [9000, 12000, 15000, 18000])
    distill.prune_checkpoints(tmp_path, keep=2)
    left = {p.name for p in tmp_path.glob("checkpoint-*")}
    assert left == {"checkpoint-15000", "checkpoint-18000"}


def test_fewer_checkpoints_than_the_limit_deletes_nothing(tmp_path):
    make_checkpoints(tmp_path, [300, 600])
    assert distill.prune_checkpoints(tmp_path, keep=3) == []
    assert len(list(tmp_path.glob("checkpoint-*"))) == 2


def test_keep_zero_or_less_disables_pruning(tmp_path):
    """A limit of 0 must mean 'keep everything', never 'delete everything'."""
    make_checkpoints(tmp_path, [300, 600, 900])
    assert distill.prune_checkpoints(tmp_path, keep=0) == []
    assert distill.prune_checkpoints(tmp_path, keep=-1) == []
    assert len(list(tmp_path.glob("checkpoint-*"))) == 3


def test_unnumbered_directories_are_never_deleted(tmp_path):
    """`output_dir` also holds the merged model and the tokenizer."""
    make_checkpoints(tmp_path, [300, 600, 900, 1200])
    keepsake = tmp_path / "checkpoint-merged"
    keepsake.mkdir()
    distill.prune_checkpoints(tmp_path, keep=1)
    assert keepsake.is_dir()
    assert (tmp_path / "checkpoint-1200").is_dir()


def test_pruning_survives_a_directory_that_vanished(tmp_path):
    """Two links of a chain can overlap for a moment at the handover."""
    make_checkpoints(tmp_path, [300, 600, 900])
    (tmp_path / "checkpoint-300" / "training_state.pt").unlink()
    (tmp_path / "checkpoint-300").rmdir()
    distill.prune_checkpoints(tmp_path, keep=1)
    assert {p.name for p in tmp_path.glob("checkpoint-*")} == {"checkpoint-900"}


def test_a_checkpoint_file_is_not_mistaken_for_a_checkpoint_dir(tmp_path):
    make_checkpoints(tmp_path, [300, 600])
    (tmp_path / "checkpoint-notes.txt").write_text("stray", encoding="utf-8")
    distill.prune_checkpoints(tmp_path, keep=1)
    assert (tmp_path / "checkpoint-notes.txt").is_file()


# ── validation ───────────────────────────────────────────────────────────
# The real pair is a 30 B MoE teacher and a 4 B student across four GPUs.
# What is worth pinning here is the contract around the forward passes:
# the cap is honoured, train mode comes back, and an empty loader is not a
# crash halfway through a 24 h job.

VOCAB = 8
SEQ = 4


class FakeOut:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Returns fixed logits and counts how many times it was called."""

    def __init__(self, fill: float):
        self.fill = fill
        self.calls = 0
        self.training = True

    def __call__(self, **batch):
        self.calls += 1
        n = batch["labels"].shape[0]
        return FakeOut(torch.full((n, SEQ, VOCAB), self.fill))

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def batches(n: int):
    for _ in range(n):
        yield {"labels": torch.zeros(1, SEQ, dtype=torch.long),
               "input_ids": torch.zeros(1, SEQ, dtype=torch.long)}


def cfg():
    return distill.DistillConfig(kl_alpha=0.7, kl_temperature=2.0)


def test_evaluate_stops_at_the_cap():
    """The reason the cap exists: the full split is over two hours per eval."""
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    distill.evaluate(student, teacher, list(batches(50)), cfg(), "cpu",
                     max_batches=7)
    assert student.calls == 7
    assert teacher.calls == 7


def test_evaluate_returns_the_three_components():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    got = distill.evaluate(student, teacher, list(batches(3)), cfg(), "cpu",
                           max_batches=3)
    assert set(got) == {"loss", "kl", "ce"}
    assert all(isinstance(v, float) for v in got.values())


def test_evaluate_restores_train_mode():
    """Left in eval mode, the rest of the run trains with dropout off."""
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    student.train()
    distill.evaluate(student, teacher, list(batches(2)), cfg(), "cpu",
                     max_batches=2)
    assert student.training is True


def test_evaluate_restores_train_mode_even_when_a_batch_raises():
    class Exploding(FakeModel):
        def __call__(self, **batch):
            raise RuntimeError("CUDA out of memory")

    student, teacher = FakeModel(0.5), Exploding(0.4)
    student.train()
    with pytest.raises(RuntimeError):
        distill.evaluate(student, teacher, list(batches(2)), cfg(), "cpu",
                         max_batches=2)
    assert student.training is True


def test_evaluate_leaves_a_model_that_was_not_training_alone():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    student.eval()
    distill.evaluate(student, teacher, list(batches(2)), cfg(), "cpu",
                     max_batches=2)
    assert student.training is False


def test_an_empty_loader_reports_nothing_rather_than_dividing_by_zero():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    assert distill.evaluate(student, teacher, [], cfg(), "cpu",
                            max_batches=10) == {}


def test_a_zero_cap_scores_nothing_and_does_not_hang():
    student, teacher = FakeModel(0.5), FakeModel(0.4)
    assert distill.evaluate(student, teacher, list(batches(5)), cfg(), "cpu",
                            max_batches=0) == {}
    assert student.calls == 0


# ── divergence guard ─────────────────────────────────────────────────────
# On 2026-09-24 the control arm went from loss 0.6255 to nan in twenty steps
# and nothing stopped it: two more links trained a dead model and saved it
# four times, and save_total_limit rotated every healthy checkpoint away. A
# 36,600-step run was lost to what should have cost one skipped update.
#
# The helpers are tested directly, and then `train()` itself is run end to
# end on CPU with stand-in models, because a guard that exists as a helper
# but is not wired into the loop is the failure mode worth ruling out.


def test_all_finite_accepts_ordinary_tensors():
    assert distill.all_finite([torch.zeros(3), torch.ones(2, 2)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_all_finite_rejects_any_non_finite_value(bad):
    t = torch.zeros(4)
    t[2] = bad
    assert not distill.all_finite([torch.zeros(3), t])


def test_all_finite_of_nothing_is_true():
    assert distill.all_finite([])


def test_no_sentinel_lets_a_run_start(tmp_path):
    assert distill.refuse_if_diverged(tmp_path) is None


def test_a_sentinel_stops_a_run_before_it_starts(tmp_path):
    (tmp_path / distill.DIVERGED_SENTINEL).write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        distill.refuse_if_diverged(tmp_path)
    assert exc.value.code == 3


def test_declaring_divergence_records_why_and_saves_nothing(tmp_path):
    with pytest.raises(SystemExit) as exc:
        distill.declare_divergence(tmp_path, 36620, "loss went to nan")
    assert exc.value.code == 3
    record = json.loads((tmp_path / distill.DIVERGED_SENTINEL).read_text(encoding="utf-8"))
    assert record["step"] == 36620
    assert "nan" in record["reason"]
    assert not list(tmp_path.glob("checkpoint-*"))


class TinyStudent(torch.nn.Module):
    """One trainable vector, so there is a real backward and a real update."""

    def __init__(self, fill: float = 0.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((VOCAB,), fill))

    def forward(self, **batch):
        n = batch["labels"].shape[0]
        return FakeOut(self.w.expand(n, SEQ, VOCAB))

    def save_pretrained(self, path, safe_serialization=True):
        torch.save(self.state_dict(), Path(path) / "student.pt")


class PoisonTeacher:
    """Uniform logits, except nan on the calls listed in `poison`."""

    def __init__(self, poison=()):
        self.poison = set(poison)
        self.calls = 0

    def __call__(self, **batch):
        i = self.calls
        self.calls += 1
        n = batch["labels"].shape[0]
        fill = float("nan") if i in self.poison else 0.1
        return FakeOut(torch.full((n, SEQ, VOCAB), fill))


class StubTokenizer:
    pad_token_id = 0
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def save_pretrained(self, *a, **k):
        pass


def run_train(monkeypatch, tmp_path, *, n_batches=12, poison=(),
              max_skipped=3, student=None):
    """Run the real `train()` on CPU with stand-ins for everything heavy."""
    teacher = PoisonTeacher(poison)
    student = student if student is not None else TinyStudent()
    loaded = {"teacher": False}

    def fake_teacher(*a, **k):
        loaded["teacher"] = True
        return teacher

    monkeypatch.setattr(distill, "AutoTokenizer", StubTokenizer)
    monkeypatch.setattr(distill, "build_dataloaders",
                        lambda cfg, tok: (list(batches(n_batches)), []))
    monkeypatch.setattr(distill, "load_teacher", fake_teacher)
    monkeypatch.setattr(distill, "load_student", lambda *a, **k: student)

    c = distill.DistillConfig(
        output_dir=str(tmp_path), student_device="cpu", bf16=False,
        gradient_checkpointing=False, gradient_accumulation_steps=2,
        logging_steps=1, eval_steps=0, save_steps=2, save_total_limit=3,
        warmup_steps=0, num_train_epochs=1, max_skipped_updates=max_skipped,
    )
    distill.train(c)
    return student, teacher, loaded


def steps_on_disk(tmp_path):
    return sorted(int(p.name.split("-")[1]) for p in tmp_path.glob("checkpoint-*"))


def test_a_clean_run_skips_nothing(monkeypatch, tmp_path, capsys):
    """For a healthy run the guard must change nothing."""
    student, _, _ = run_train(monkeypatch, tmp_path)
    assert "[guard]" not in capsys.readouterr().err
    assert steps_on_disk(tmp_path) == [2, 4, 6]       # 12 batches / 2 = 6
    assert not (tmp_path / distill.DIVERGED_SENTINEL).exists()
    assert distill.all_finite(student.parameters())


def test_one_poisoned_batch_costs_one_update_not_the_run(monkeypatch, tmp_path,
                                                         capsys):
    """The case that killed the control arm: a single bad batch."""
    student, _, _ = run_train(monkeypatch, tmp_path, poison={5})
    err = capsys.readouterr().err
    assert "skipped an update" in err
    # Window 3 (batches 4 and 5) is dropped: five updates instead of six.
    assert steps_on_disk(tmp_path)[-1] == 5
    assert distill.all_finite(student.parameters())
    assert not (tmp_path / distill.DIVERGED_SENTINEL).exists()


def test_a_run_that_stays_nan_stops_and_keeps_the_healthy_checkpoints(
        monkeypatch, tmp_path):
    """The other half: a real divergence is stopped, and stopped WITHOUT
    saving — so the checkpoint written before it survives the rotation that
    destroyed the control arm's."""
    with pytest.raises(SystemExit) as exc:
        run_train(monkeypatch, tmp_path, poison=set(range(4, 100)),
                  max_skipped=3)
    assert exc.value.code == 3
    record = json.loads((tmp_path / distill.DIVERGED_SENTINEL).read_text(encoding="utf-8"))
    assert record["step"] == 2
    assert steps_on_disk(tmp_path) == [2]
    saved = torch.load(tmp_path / "checkpoint-2" / "student.pt")
    assert distill.all_finite(saved.values())


def test_a_diverged_chain_refuses_before_loading_the_teacher(monkeypatch,
                                                             tmp_path):
    """Every later link of the afterany chain must cost seconds, not a
    61 GB teacher load."""
    (tmp_path / distill.DIVERGED_SENTINEL).write_text('{"step": 36620}\n', encoding="utf-8")
    loaded = {}
    with pytest.raises(SystemExit) as exc:
        _, _, loaded = run_train(monkeypatch, tmp_path)
    assert exc.value.code == 3
    assert loaded == {}          # run_train never returned: nothing loaded


def test_resuming_from_a_non_finite_checkpoint_is_caught_at_load(monkeypatch,
                                                                 tmp_path):
    """A checkpoint already holding nan would make every step nan; catching
    it at load costs one model load instead of the whole link."""
    # A checkpoint that finished being written: latest_checkpoint skips one
    # with no training_state.pt (a write cut short), and this test is about
    # the nan, not about a half-written directory.
    (tmp_path / "checkpoint-4").mkdir()
    write_state(tmp_path / "checkpoint-4" / "training_state.pt")
    monkeypatch.setattr(distill, "_reload_student_from_checkpoint",
                        lambda *a, **k: TinyStudent(float("nan")))
    monkeypatch.setattr(distill, "load_checkpoint", lambda *a, **k: 4)
    with pytest.raises(SystemExit) as exc:
        run_train(monkeypatch, tmp_path)
    assert exc.value.code == 3
    record = json.loads((tmp_path / distill.DIVERGED_SENTINEL).read_text(encoding="utf-8"))
    assert record["step"] == 4
    assert "checkpoint-4" in record["reason"]


# --- a resumed run continues the data where it stopped ---------------------------
#
# Regression for the split arm, 25 September: every resume started the data
# from the top, so a chain of 2-hour links retrained the same ~650 steps of
# corpus a dozen times and held-out perplexity went 5.11 -> 5.92.

class RecordingTeacher:
    """Uniform logits; remembers which batch (by its id) it was shown."""

    def __init__(self):
        self.seen = []

    def __call__(self, **batch):
        self.seen.append(int(batch["input_ids"][0, 0]))
        n = batch["labels"].shape[0]
        return FakeOut(torch.full((n, SEQ, VOCAB), 0.1))


def numbered_batches(n: int):
    return [{"labels": torch.zeros(1, SEQ, dtype=torch.long),
             "input_ids": torch.full((1, SEQ), i, dtype=torch.long)}
            for i in range(n)]


def run_link(monkeypatch, tmp_path, *, n_batches=12, max_steps=-1):
    """One link of a chain: train() on CPU, resuming from tmp_path if it can."""
    teacher = RecordingTeacher()

    def reload(ckpt_dir, cfg, dtype, device):
        student = TinyStudent()
        student.load_state_dict(torch.load(Path(ckpt_dir) / "student.pt"))
        return student

    monkeypatch.setattr(distill, "AutoTokenizer", StubTokenizer)
    monkeypatch.setattr(distill, "build_dataloaders",
                        lambda cfg, tok: (numbered_batches(n_batches), []))
    monkeypatch.setattr(distill, "load_teacher", lambda *a, **k: teacher)
    monkeypatch.setattr(distill, "load_student", lambda *a, **k: TinyStudent())
    monkeypatch.setattr(distill, "_reload_student_from_checkpoint", reload)

    distill.train(distill.DistillConfig(
        output_dir=str(tmp_path), student_device="cpu", bf16=False,
        gradient_checkpointing=False, gradient_accumulation_steps=2,
        logging_steps=1, eval_steps=0, save_steps=1, save_total_limit=10,
        warmup_steps=0, num_train_epochs=1, max_steps=max_steps,
    ))
    return teacher.seen


def test_a_resumed_link_continues_the_data_instead_of_restarting_it(monkeypatch, tmp_path):
    # First link stops after 2 optimizer steps = 4 micro-batches (0-3).
    first = run_link(monkeypatch, tmp_path, max_steps=2)
    assert first == [0, 1, 2, 3]

    # The next link resumes at step 2 and must go on from batch 4. Before the
    # fix it saw [0, 1, 2, …] again — the same data as the link before it.
    second = run_link(monkeypatch, tmp_path)
    assert second == list(range(4, 12))


def test_a_resumed_run_stops_at_its_schedule_not_after_another_epoch(monkeypatch, tmp_path):
    run_link(monkeypatch, tmp_path, max_steps=2)
    run_link(monkeypatch, tmp_path)
    # 12 batches / 2 per step = 6 steps in the epoch, and not one more.
    assert max(steps_on_disk(tmp_path)) == 6

    # A link submitted after the run is complete trains on nothing.
    assert run_link(monkeypatch, tmp_path) == []


def test_data_position_counts_micro_batches_across_epochs():
    assert distill.data_position(0, 8, 100) == (0, 0)
    assert distill.data_position(10, 8, 100) == (0, 80)
    assert distill.data_position(25, 8, 100) == (2, 0)
    assert distill.data_position(26, 8, 100) == (2, 8)


def test_a_real_loader_has_one_fixed_order_per_epoch_and_skips_by_batch():
    """The permutation depends on (seed, epoch) only, so every link agrees on it."""
    loader = torch.utils.data.DataLoader(list(range(20)), batch_size=4, shuffle=True)
    a = [b.tolist() for b in distill.epoch_batches(loader, epoch=0, skip=0, seed=7)]
    torch.manual_seed(12345)          # whatever else consumed the global RNG
    b = [b.tolist() for b in distill.epoch_batches(loader, epoch=0, skip=0, seed=7)]
    assert a == b
    assert sorted(x for batch in a for x in batch) == list(range(20))

    rest = [b.tolist() for b in distill.epoch_batches(loader, epoch=0, skip=2, seed=7)]
    assert rest == a[2:]

    other = [b.tolist() for b in distill.epoch_batches(loader, epoch=1, skip=0, seed=7)]
    assert other != a


def test_the_tokenised_cache_fingerprint_is_stable_and_specific(tmp_path):
    """Every link of every chain used to miss the tokenised cache and write a
    fresh copy (1,107 files, 539 GB in the shared HF cache on 2026-09-27).
    The fingerprint is now built from what the result depends on, so it is the
    same on every restart and different when any of those changes."""
    import types

    data = tmp_path / "train.jsonl"
    data.write_text('{"text": "a"}\n', encoding="utf-8")
    files = {"train": str(data), "validation": str(data)}
    cfg = types.SimpleNamespace(cutoff_len=2048, max_train_samples=None)

    class Tok:
        name_or_path = "Qwen/Qwen3-4B-Base"

        def __len__(self):
            return 151669

    fp = distill.tokenize_fingerprint
    base = fp(cfg, Tok(), files, "train")
    assert base == fp(cfg, Tok(), files, "train")                 # same on restart
    assert base != fp(cfg, Tok(), files, "validation")            # per split
    assert base != fp(types.SimpleNamespace(cutoff_len=1024, max_train_samples=None),
                      Tok(), files, "train")                      # cutoff
    other = Tok()
    other.name_or_path = "Qwen/Qwen3-8B-Base"
    assert base != fp(cfg, other, files, "train")                 # tokenizer
    data.write_text('{"text": "a rebuilt corpus"}\n', encoding="utf-8")
    assert base != fp(cfg, Tok(), files, "train")                 # rebuilt corpus


# ── the teacher's inputs ─────────────────────────────────────────────────
# JUPITER (15-10-2026): a teacher whole on one GPU has no accelerate hooks to
# move a batch that sits on the student's GPU, and the first forward died.

class Moved:
    """A stand-in tensor that records where it was sent."""

    def __init__(self):
        self.to_device = None

    def to(self, device, non_blocking=False):
        self.to_device = device
        return self


class WithEmbeddings:
    def __init__(self, device):
        self.emb = torch.nn.Embedding(VOCAB, 2, device=device)

    def get_input_embeddings(self):
        return self.emb


def test_the_batch_goes_to_the_device_of_the_teachers_embeddings():
    batch = {"input_ids": Moved(), "attention_mask": Moved(), "labels": Moved()}
    out = distill.teacher_batch(WithEmbeddings("cpu"), batch)
    assert {k: v.to_device for k, v in out.items()} == {k: torch.device("cpu") for k in batch}


def test_a_teacher_on_the_meta_device_keeps_the_batch_for_its_hooks():
    batch = {"input_ids": Moved()}
    out = distill.teacher_batch(WithEmbeddings("meta"), batch)
    assert out is batch and batch["input_ids"].to_device is None


def test_a_teacher_without_input_embeddings_keeps_the_batch():
    batch = {"input_ids": Moved()}
    assert distill.teacher_batch(FakeModel(0.0), batch) is batch


# ── packed sequences ─────────────────────────────────────────────────────
# JUPITER's legislation chunks are ~210 tokens: one a micro-batch ran at ~340
# tokens/s. Packed, several share a sequence, and must not see one another.

def test_documents_are_packed_up_to_the_cutoff_each_from_position_zero():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
    packs = distill.pack_examples(docs, cutoff=6, eos_id=None, seed=0)
    assert all(len(p["input_ids"]) <= 6 for p in packs)
    assert sorted(t for p in packs for t in p["input_ids"]) == list(range(1, 11))
    for p in packs:
        assert len(p["input_ids"]) == len(p["position_ids"]) == len(p["labels"])
        starts = [i for i, pos in enumerate(p["position_ids"]) if pos == 0]
        assert starts[0] == 0
        # The first token of every later document is not a target.
        assert all(p["labels"][i] == -100 for i in starts[1:])
        assert all(p["labels"][i] == p["input_ids"][i]
                   for i in range(len(p["labels"])) if i not in starts[1:])


def test_each_document_ends_with_eos_and_a_long_one_is_cut():
    packs = distill.pack_examples([[1, 2, 3, 4, 5, 6, 7, 8]], cutoff=4, eos_id=99, seed=0)
    assert packs == [{"input_ids": [1, 2, 3, 4], "position_ids": [0, 1, 2, 3],
                      "labels": [1, 2, 3, 4]}]
    packs = distill.pack_examples([[1, 2], [3]], cutoff=10, eos_id=99, seed=0)
    assert sum(p["input_ids"].count(99) for p in packs) == 2


def test_the_packing_is_the_same_for_the_same_seed():
    docs = [[i] * (i % 5 + 1) for i in range(1, 40)]
    assert (distill.pack_examples(docs, 8, None, seed=3)
            == distill.pack_examples(docs, 8, None, seed=3))


def test_a_packed_batch_has_no_attention_mask_and_takes_one_sequence():
    row = {"input_ids": [1, 2], "position_ids": [0, 1], "labels": [1, 2]}
    out = distill.collate_packed([row])
    assert set(out) == {"input_ids", "position_ids", "labels"}
    assert out["input_ids"].shape == (1, 2)
    with pytest.raises(ValueError):
        distill.collate_packed([row, row])


def test_a_packed_document_sees_only_itself():
    """Logits of a document packed after another equal its logits alone,
    in a real (tiny) Qwen3: transformers' packed-sequence mask is in effect.
    Without use_cache=False the cache turns it off, and they differ."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128)).eval()
    a, b = [5, 9, 3, 7, 11], [20, 21, 22, 23]
    pack = distill.collate_packed(distill.pack_examples([a, b], 16, None, seed=0))
    ids = pack["input_ids"][0].tolist()
    start = len(ids) - len(b)
    assert ids[start:] == b and start > 0  # b after a: the case that can leak
    cfg = distill.DistillConfig(pack_sequences=True)
    with torch.no_grad():
        packed = model(input_ids=pack["input_ids"], position_ids=pack["position_ids"],
                       **distill.model_kwargs(cfg)).logits[0, start:start + len(b)]
        alone = model(input_ids=torch.tensor([b]), use_cache=False).logits[0]
        leaky = model(input_ids=pack["input_ids"],
                      position_ids=pack["position_ids"]).logits[0, start:start + len(b)]
    torch.testing.assert_close(packed, alone, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(leaky, alone, atol=1e-4)  # with a cache, b sees a
