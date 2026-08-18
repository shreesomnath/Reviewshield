"""DPO fine-tuning of the reviewer model (D2, plan.md sec:5/7, Days 9-12).

Trains a LoRA adapter on the primary reviewer base
(Qwen2.5-14B-Instruct by default; pass --base-model-id for the secondary
Llama-3.1-8B-Instruct robustness check) against the preference pairs from
generate_preference_pairs.py. Uses Unsloth for fast/low-memory LoRA + TRL's
DPOTrainer for the actual DPO loss.

Every stored pair's `prompt` field is already the neutral (undefended)
framing (see review_prompt.py / generate_preference_pairs.py docstrings),
so this script trains the model to resist injected instructions WITHOUT
being told to at inference time - at evaluation, D0 (base) and D2 (this
model) are queried with the identical neutral prompt; only the weights
differ.

Usage (inside the container):
    python /workspace/scripts/training/train_dpo.py \
        --base-model-id Qwen/Qwen2.5-14B-Instruct \
        --train-pairs /workspace/data/processed/preference_pairs/train_pairs.jsonl \
        --val-pairs /workspace/data/processed/preference_pairs/val_pairs.jsonl \
        --out-dir /workspace/outputs/checkpoints/dpo_qwen14b
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# unsloth must be imported before trl/transformers/peft, or it warns that
# its optimizations may not fully apply (observed directly on the first
# real training run - not a guess).
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

DEFAULT_BASE = "Qwen/Qwen2.5-14B-Instruct"


def load_pairs_dataset(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    # DPOTrainer (TRL >=0.12) expects exactly these three columns.
    return ds.remove_columns(
        [c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")]
    )


def main(args):
    print(f"Loading base model for LoRA/DPO: {args.base_model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,  # let unsloth pick bf16 on this hardware
        load_in_4bit=False,  # full-precision LoRA base for training quality;
                              # 96 GB per GPU comfortably fits a 14B model
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_ds = load_pairs_dataset(args.train_pairs)
    val_ds = load_pairs_dataset(args.val_pairs) if args.val_pairs else None
    print(f"Train pairs: {len(train_ds)}" + (f"  Val pairs: {len(val_ds)}" if val_ds else ""))

    config = DPOConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=int(args.max_seq_length * 0.7),
        logging_steps=10,
        # Periodic checkpointing, fixed properly this time. The original
        # PicklingError (dry run) came from TRL's default save trying to
        # pickle the full trainer state (optimizer/scheduler) under
        # unsloth's dynamically-patched DPOConfig class. save_only_model=True
        # skips that entirely and writes just the model/adapter weights via
        # safetensors, which sidesteps the picklability issue. Previously
        # this was set to "no" with no periodic saves at all, which meant
        # the real run that OOM'd at step 133/266 lost 100% of its progress
        # with nothing to resume from - never doing that again.
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=True,
        save_total_limit=3,
        # Disabled: the real run's OOM crash happened inside this epoch-end
        # eval pass (confirmed via the traceback - accelerate's fp32
        # conversion during DPOTrainer.evaluate()), on top of training
        # already using ~94/95 GB. We don't need periodic validation loss;
        # the actual quality measure for this project is the post-training
        # D0-vs-D2 comparison from run_eval.py, not DPO loss during training.
        eval_strategy="no",
        bf16=True,
        report_to=[],
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    final_dir = f"{args.out_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-id", default=DEFAULT_BASE)
    ap.add_argument("--train-pairs", required=True)
    ap.add_argument("--val-pairs", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq-length", type=int, default=6000)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=40,
                     help="save a checkpoint every N steps (~10-15 min of "
                          "compute at a time, so a crash never loses more "
                          "than that again)")
    main(ap.parse_args())
