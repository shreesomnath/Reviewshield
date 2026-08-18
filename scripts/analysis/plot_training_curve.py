"""DPO training-dynamics figure, built directly from the real training log
(logs/train_dpo_real_v2.log) - no need to wait for evaluation to finish,
since this only needs data training itself already produced.

Plots three panels over training progress (logging_steps=10, so each
logged point is 10 real optimizer steps):
  - DPO loss
  - reward accuracy (fraction of pairs where the model already prefers
    the resistant/chosen response over the compromised/rejected one)
  - reward margin (how confidently chosen is preferred over rejected)

Usage (inside the container, or with matplotlib available locally):
    python /workspace/scripts/analysis/plot_training_curve.py
"""
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})
import matplotlib.pyplot as plt

LOG_PATH = Path("/workspace/logs/train_dpo_real_v2.log")
FIG_DIR = Path("/workspace/outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_log(log_path: Path) -> list[dict]:
    text = log_path.read_text()
    records = []
    for m in re.finditer(r"\{'loss':[^}]+\}", text):
        try:
            records.append(ast.literal_eval(m.group(0)))
        except (ValueError, SyntaxError):
            continue
    return records


def main():
    records = parse_log(LOG_PATH)
    print(f"Parsed {len(records)} logged training steps from {LOG_PATH}")
    if not records:
        print("No records found - nothing to plot.")
        return

    epochs = [r["epoch"] for r in records]
    loss = [r["loss"] for r in records]
    acc = [r["rewards/accuracies"] * 100 for r in records]
    margin = [r["rewards/margins"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, loss, marker="o", markersize=3, color="tab:red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("DPO loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, acc, marker="o", markersize=3, color="tab:blue")
    axes[1].axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="chance (50%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reward accuracy (%)")
    axes[1].set_title("Fraction preferring resistant\nover compromised response")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, margin, marker="o", markersize=3, color="tab:green")
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Reward margin")
    axes[2].set_title("Confidence margin\n(chosen - rejected reward)")
    axes[2].grid(alpha=0.3)

    fig.suptitle("D2 (DPO) training dynamics — Qwen2.5-14B-Instruct, "
                  f"{len(records)} logged steps over {epochs[-1]:.1f} epochs", y=1.03)
    fig.tight_layout()
    out_path = FIG_DIR / "training_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    print(f"\nSummary: loss {loss[0]:.3f} -> {loss[-1]:.3f}, "
          f"reward accuracy {acc[0]:.1f}% -> {acc[-1]:.1f}%, "
          f"reward margin {margin[0]:.3f} -> {margin[-1]:.3f}")


if __name__ == "__main__":
    main()
