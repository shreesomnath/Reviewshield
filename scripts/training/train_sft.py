"""SFT-only ablation (D3, plan.md sec:5/7): trains a LoRA adapter with
plain supervised fine-tuning on ONLY the preferred ("chosen") response
from each preference pair -- no contrastive/rejected signal at all --
to isolate whether DPO's preference-comparison objective is doing
something SFT-on-good-examples alone would not.

Same base model, same LoRA rank/target-modules, and the same neutral
(undefended) prompt as train_dpo.py, so D2 (DPO) and D3 (this script) are
a clean ablation of ONE variable: the training objective. Everything else
-- data, base model, adapter capacity -- is held fixed.

Learning rate deliberately does NOT reuse DPO's 5e-6: that value is
calibrated for DPO's beta-scaled preference loss and would badly
undertrain a plain next-token SFT objective, which would make D3 look
worse than it really is for a reason having nothing to do with the
ablation question. 2e-4 is the standard Unsloth-documented LoRA SFT
learning rate.

Usage (inside the container):
    python /workspace/scripts/training/train_sft.py \
        --base-model-id Qwen/Qwen2.5-14B-Instruct \
        --train-pairs /workspace/data/processed/preference_pairs/train_pairs.jsonl \
        --val-pairs /workspace/data/processed/preference_pairs/val_pairs.jsonl \
        --out-dir /workspace/outputs/checkpoints/sft_qwen14b_v1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

DEFAULT_BASE = "Qwen/Qwen2.5-14B-Instruct"


def load_sft_dataset(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.rename_column("chosen", "completion")
    return ds.remove_columns(
        [c for c in ds.column_names if c not in ("prompt", "completion")]
    )


def main(args):
    print(f"Loading base model for LoRA/SFT: {args.base_model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
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

    train_ds = load_sft_dataset(args.train_pairs)
    val_ds = load_sft_dataset(args.val_pairs) if args.val_pairs else None
    print(f"Train examples: {len(train_ds)}" + (f"  Val examples: {len(val_ds)}" if val_ds else ""))

    config = SFTConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        # Same checkpointing lesson as train_dpo.py (process.md Step 23):
        # save_only_model=True avoids the PicklingError under Unsloth's
        # dynamically-patched config classes, periodic steps-based saving
        # means a crash never loses more than ~10-15 min of progress.
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=True,
        save_total_limit=3,
        eval_strategy="no",
        bf16=True,
        report_to=[],
        seed=args.seed,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
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
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=40)
    main(ap.parse_args())
