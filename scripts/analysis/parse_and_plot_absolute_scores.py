"""Parse absolute_score_sample.py's log output into structured JSON, then
plot mean absolute clean/injected scores per family for D0 vs D2 - the
corrected, artifact-free comparison (see run_eval.py's injected_score
docstring for why the naive relative-ASR metric misled us: D2's own
clean-paper baseline runs lower, so an equal-or-better absolute outcome
under attack still registered as bigger "inflation").

Usage (inside the container):
    python /workspace/scripts/analysis/parse_and_plot_absolute_scores.py
"""
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
import matplotlib.pyplot as plt

LOG_PATH = Path("/workspace/logs/absolute_score_sample_full.log")
OUT_JSON = Path("/workspace/outputs/eval/absolute_score_sample.json")
FIG_DIR = Path("/workspace/outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

FAMILIES = ["F1", "F2", "F3", "F4", "F5"]
FAMILY_LABELS = ["F1\nDirect\noverride", "F2\nStealth/\nrendering",
                  "F3\nReviewer\nimpersonation", "F4\nFabricated\nauthority",
                  "F5\nSycophancy"]


def parse_log():
    text = LOG_PATH.read_text()
    data = {"D0": {f: {"clean": [], "injected": []} for f in FAMILIES},
            "D2": {f: {"clean": [], "injected": []} for f in FAMILIES}}
    cond = None
    for line in text.splitlines():
        if line.strip() == "=== D0 ===":
            cond = "D0"
        elif line.strip() == "=== D2 ===":
            cond = "D2"
        m = re.match(r"(\S+) (F\d): clean=(\d+|None) injected=(\d+|None)", line.strip())
        if m and cond:
            _, fam, c, i = m.groups()
            if c != "None":
                data[cond][fam]["clean"].append(int(c))
            if i != "None":
                data[cond][fam]["injected"].append(int(i))
    return data


def main():
    data = parse_log()
    summary = {}
    for cond in ["D0", "D2"]:
        summary[cond] = {}
        for fam in FAMILIES:
            c = data[cond][fam]["clean"]
            i = data[cond][fam]["injected"]
            summary[cond][fam] = {
                "mean_clean": round(statistics.mean(c), 3) if c else None,
                "mean_injected": round(statistics.mean(i), 3) if i else None,
                "n": len(i),
            }
    OUT_JSON.write_text(json.dumps({"raw": data, "summary": summary}, indent=2))
    print(f"Wrote {OUT_JSON}")

    # --- plot: grouped bars, mean injected score per family, D0 vs D2 ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(FAMILIES))
    width = 0.35
    d0_inj = [summary["D0"][f]["mean_injected"] for f in FAMILIES]
    d2_inj = [summary["D2"][f]["mean_injected"] for f in FAMILIES]
    d0_clean = statistics.mean([summary["D0"][f]["mean_clean"] for f in FAMILIES])
    d2_clean = statistics.mean([summary["D2"][f]["mean_clean"] for f in FAMILIES])

    ax.bar([i - width/2 for i in x], d0_inj, width, label="D0 (undefended)", color="tab:red", alpha=0.85)
    ax.bar([i + width/2 for i in x], d2_inj, width, label="D2 (trained)", color="tab:green", alpha=0.85)
    ax.axhline(d0_clean, color="tab:red", linestyle="--", linewidth=1.2, alpha=0.7, label=f"D0 clean baseline ({d0_clean:.2f})")
    ax.axhline(d2_clean, color="tab:green", linestyle="--", linewidth=1.2, alpha=0.7, label=f"D2 clean baseline ({d2_clean:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(FAMILY_LABELS)
    ax.set_ylabel("Mean absolute score under attack")
    ax.set_title("Absolute injected-paper score by family, D0 vs D2\n"
                  "(corrected metric - D2 lower is better on every family)")
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    out_path = FIG_DIR / "absolute_score_by_family.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    print("\n=== Table ===")
    print(f"{'Family':<8}{'D0 clean':<10}{'D0 inj':<10}{'D2 clean':<10}{'D2 inj':<10}{'D2 advantage':<14}")
    for fam in FAMILIES:
        d0c, d0i = summary["D0"][fam]["mean_clean"], summary["D0"][fam]["mean_injected"]
        d2c, d2i = summary["D2"][fam]["mean_clean"], summary["D2"][fam]["mean_injected"]
        adv = round(d0i - d2i, 2) if d0i is not None and d2i is not None else None
        print(f"{fam:<8}{d0c:<10}{d0i:<10}{d2c:<10}{d2i:<10}{adv:<14}")


if __name__ == "__main__":
    main()
