#!/usr/bin/env python3
"""GRPO with verifiable rewards on an open-book model, as a LoRA adapter.

    accelerate launch --num_processes 4 forge/scripts/grpo_train.py \\
        --model  $WORK/eullm_runs/stage3/sft-v03-instruct-ob/merged \\
        --prompts $WORK/eullm_runs/grpo/prompts.jsonl \\
        --out    $WORK/eullm_runs/grpo/v03

For each prompt the model writes ``--num-generations`` answers; each is
scored by `eullm_forge.rl.answer_reward` (1 right, 0 wrong), and the update
favours the answers that beat their group's mean. Starting from a model that
already answers in the open-book format (a merged stage-3 model), not from a
base model: RL sharpens what the model can already do sometimes, it does not
teach a format from nothing.

With ``--judge-url`` the questions on what an article provides
("contenuto", make_grpo_prompts.py --judged) are graded by a judge model
behind that URL (`eullm_forge.rl.judge_reward`); without it a prompts file
holding such questions is refused, since nothing could score them.

The adapter lands in ``--out/adapter`` like stage 3's, so
leonardo/sbatch_stage3_package.slurm merges and packages it unchanged, with
``S3_BASE`` the model given here as ``--model``.

Resumable: a rerun with the same --out continues from its last checkpoint,
and one whose adapter is already there stops at once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eullm_forge.identity import load_text_model, lora_target_modules  # noqa: E402
from eullm_forge.rl import JUDGED_TYPES, answer_reward  # noqa: E402
from eullm_forge.rl.guard import RunGuard  # noqa: E402


def load_prompts(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit(f"[grpo] no prompts in {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged open-book model to start from")
    ap.add_argument("--prompts", required=True, type=Path, help="make_grpo_prompts.py output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rank", type=int, default=32)
    # RL moves a model that already works; a stage-3 learning rate would
    # undo the format it learnt there in a few dozen steps.
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=8,
                    help="completions per GPU per step; "
                         "prompts per step = this x GPUs / generations")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-completion", type=int, default=384,
                    help="the exam allows 300 new tokens; a little room, not a license to ramble")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.02,
                    help="KL to the starting model: keeps it from drifting off what it could do")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--judge-url", help="OpenAI-compatible chat endpoint of the judge that "
                    "grades the judged questions (e.g. http://127.0.0.1:8080)")
    ap.add_argument("--judge-parallel", type=int, default=8,
                    help="grading requests sent to the judge at once")
    ap.add_argument("--citation-weight", type=float, default=0.0,
                    help="weight of eullm_forge.rl.citation_reward, paid to a judged answer "
                         "that cites its question's article (0: off, as on Leonardo)")
    ap.add_argument("--seed", type=int, default=42,
                    help="order of the prompts: a round continuing another on the same "
                         "prompts takes another seed, or it sees the same ones first")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N prompts (smoke tests)")
    args = ap.parse_args(argv)

    done = args.out / "adapter" / "adapter_config.json"
    if done.is_file() and done.stat().st_size > 0:
        print(f"[grpo] adapter already at {done.parent}: nothing left to do "
              "(move it away to train again)", flush=True)
        return 0

    rows = load_prompts(args.prompts)
    if args.limit:
        rows = rows[:args.limit]
    judged = sum(r["tipo"] in JUDGED_TYPES for r in rows)
    if judged and not args.judge_url:
        raise SystemExit(f"[grpo] {judged} prompts need a judge: give --judge-url")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    # Every row has every column (a dataset is a table); the judge's three
    # are empty on the rows it does not grade.
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "tipo": r["tipo"], "keywords": r["keywords"],
          "question": r.get("question", ""), "reference": r.get("reference", ""),
          "rubric": r.get("rubric", ""), "code": str(r.get("code") or ""),
          "articolo": str(r.get("articolo") or "")} for r in rows])
    rewards = [answer_reward]
    if args.judge_url:
        from eullm_forge.rl.judge_reward import JudgeReward
        rewards.append(JudgeReward(args.judge_url, parallel=args.judge_parallel))
        print(f"[grpo] {judged} of {len(rows)} prompts graded by the judge at "
              f"{args.judge_url}", flush=True)
    weights = None
    if args.citation_weight > 0:
        from eullm_forge.rl.citation import citation_reward
        rewards.append(citation_reward)
        weights = [1.0] * (len(rewards) - 1) + [args.citation_weight]
        print(f"[grpo] citing the question's article pays {args.citation_weight}", flush=True)

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_text_model(args.model, dtype=torch.bfloat16 if cuda else torch.float32)

    config = GRPOConfig(
        output_dir=str(args.out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        max_steps=args.max_steps,
        beta=args.beta,
        temperature=args.temperature,
        # The exam and the engine ask with thinking off (legal_eval.chat_prompt).
        chat_template_kwargs={"enable_thinking": False},
        bf16=cuda,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        seed=args.seed,
        reward_weights=weights,
        use_cpu=not cuda,
    )
    peft_config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                             target_modules=lora_target_modules(model), task_type="CAUSAL_LM")
    from transformers import TrainerCallback

    guard = RunGuard(max_steps=args.max_steps)

    class Guard(TrainerCallback):
        """Prints progress and stops a run whose signal is broken or gone."""

        def on_log(self, _args, state, control, logs=None, **_):
            logs = logs or {}
            line = guard.progress(state.global_step, logs)
            if line and state.is_world_process_zero:
                print(line, flush=True)
            if guard.observe(state.global_step, logs):
                control.should_training_stop = True

    trainer = GRPOTrainer(model=model, reward_funcs=rewards, args=config,
                          train_dataset=dataset, processing_class=tok, peft_config=peft_config,
                          callbacks=[Guard()])

    last = get_last_checkpoint(str(args.out)) if args.out.is_dir() else None
    if last:
        print(f"[grpo] resuming from {last}", flush=True)
    print(f"[grpo] {len(dataset)} prompts, {args.num_generations} answers each, "
          f"up to {args.max_steps} steps", flush=True)
    trainer.train(resume_from_checkpoint=last)

    if guard.stopped and trainer.accelerator.is_main_process:
        print(f"[grpo] STOP: {guard.stopped}", flush=True)
        (args.out / "STOPPED").write_text(guard.stopped + "\n", encoding="utf-8")
    if guard.broken:
        # No adapter: broken weights are not a model to package, and their
        # absence keeps the package and exam jobs behind this one from
        # grading one. The exit status makes the job FAILED, which status.sh
        # flags along with the STOP line. A saturated run falls through and
        # saves what it learnt.
        return 1

    if trainer.accelerator.is_main_process:
        trainer.model.save_pretrained(str(args.out / "adapter"))
        tok.save_pretrained(str(args.out / "adapter"))
        print(f"[grpo] adapter {args.out / 'adapter'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
