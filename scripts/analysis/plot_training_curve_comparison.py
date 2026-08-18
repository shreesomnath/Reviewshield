"""Cross-model DPO training-dynamics comparison: Qwen2.5-14B vs
Llama-3.1-8B, both real, complete training logs.

Usage (inside the container):
    python /workspace/scripts/analysis/plot_training_curve_comparison.py
"""
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
import matplotlib.pyplot as plt

LOGS = {
    "Qwen2.5-14B": Path("/workspace/logs/train_dpo_real_v2.log"),
    "Llama-3.1-8B": Path("/workspace/logs/train_dpo_llama_real.log"),
}
COLORS = {"Qwen2.5-14B": "tab:blue", "Llama-3.1-8B": "tab:orange"}
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
    data = {label: parse_log(path) for label, path in LOGS.items()}
    for label, records in data.items():
        print(f"{label}: {len(records)} logged steps")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for label, records in data.items():
        epochs = [r["epoch"] for r in records]
        loss = [r["loss"] for r in records]
        acc = [r["rewards/accuracies"] * 100 for r in records]
        margin = [r["rewards/margins"] for r in records]
        c = COLORS[label]

        axes[0].plot(epochs, loss, marker="o", markersize=3, color=c, label=label)
        axes[1].plot(epochs, acc, marker="o", markersize=3, color=c, label=label)
        axes[2].plot(epochs, margin, marker="o", markersize=3, color=c, label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("DPO loss")
    axes[0].set_title("Training loss"); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Reward accuracy (%)")
    axes[1].set_title("Fraction preferring resistant\nover compromised response")
    axes[1].grid(alpha=0.3); axes[1].legend()

    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Reward margin")
    axes[2].set_title("Confidence margin\n(chosen - rejected reward)")
    axes[2].grid(alpha=0.3); axes[2].legend()

    fig.suptitle("DPO training dynamics: Qwen2.5-14B vs Llama-3.1-8B", y=1.03)
    fig.tight_layout()
    out_path = FIG_DIR / "training_curve_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    for label, records in data.items():
        loss = [r["loss"] for r in records]
        acc = [r["rewards/accuracies"] * 100 for r in records]
        print(f"{label}: loss {loss[0]:.3f}->{loss[-1]:.3f}, "
              f"accuracy {acc[0]:.1f}%->{acc[-1]:.1f}%")


if __name__ == "__main__":
    main()
