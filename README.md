# ReviewShield

Code and benchmark for **"ReviewShield: Defending LLM Reviewers Against In-Paper Prompt Injection under Instruction-Content Entanglement"** by Somnath Luitel.

---

An author who controls their own manuscript can embed hidden instructions engineered to manipulate whichever LLM reviews it. Defending against this is harder than generic prompt injection because of what we call **instruction-content entanglement**: the manuscript is at once the untrusted channel carrying the attack and the legitimate object the reviewer must engage with critically. A defense can't just learn to ignore instruction-like text without also breaking the review itself.

This repository contains **ReviewShield-Bench**, a benchmark of five in-paper injection families embedded in real open-access manuscripts. It includes the training pipeline (DPO, a prompting defense, and an SFT ablation, across three reviewer models), the evaluation harness, and the paired difference-in-differences (DID) statistical protocol introduced to correctly measure a defense's effect once its own scoring calibration is accounted for.

---

## 📂 Repository Layout

```text
scripts/
  download_arxiv_corpus.py      Pull open-access CS/ML papers from arXiv
  audit_corpus.py               Content-quality audit of the pulled corpus
  download_models.py            Pull the HF model weights this project uses
  benchmark_assembly.py         Assemble ReviewShield-Bench (splits + variants)
  flaws.py                      Deterministic planted-flaw templates
  inject.py                     Splice an attack into a manuscript
  review_prompt.py              The shared reviewer prompt
  smoke_test_loop.py            End-to-end parse -> inject -> generate smoke test
  attacks/                      F1-F5 attack families
  generation/                   Model loading, API clients, and dataset generation
  training/                     DPO and SFT training scripts
  evaluation/                   Main harness and evaluation scripts
  analysis/                     DID protocol, statistical analysis, and plotting

dev_scripts/
  One-off diagnostic, debugging, and historical-provenance scripts.

environment/
  revguard.def                  Apptainer/Singularity container definition.

run/
  Shell scripts used to run the pipeline stages.
```

---

## 🛠️ Environment

Everything runs inside the Apptainer container defined in `environment/revguard.def` (PyTorch 2.7.0 + CUDA 12.8, Python 3.11).

```bash
apptainer build --fakeroot --nv revguard.sif environment/revguard.def
apptainer exec --nv --bind /path/to/project:/workspace revguard.sif \
    python /workspace/scripts/<script>.py
```

Credentials are passed at runtime, never baked into the image:

```bash
apptainer exec --env HF_TOKEN=... --env GEMINI_API_KEY=... --nv \
    --bind /path/to/project:/workspace revguard.sif python ...
```

---

## 🚀 Reproducing the Pipeline End to End

1. **Corpus.** `download_arxiv_corpus.py` -> `audit_corpus.py` -> `benchmark_assembly.py`. Produces ReviewShield-Bench.
2. **Preference pairs.** `generation/generate_preference_pairs.py` generates two candidate reviews per injected training-split manuscript.
3. **Training.** `training/train_dpo.py` and `training/train_sft.py`.
4. **Evaluation.** `evaluation/run_eval.py --condition {D0,D1,D2,D3}` runs the main harness.
5. **Statistics.** `analysis/did_analysis.py` computes the central metric.
6. **Figures.** Scripts under `analysis/plot_*.py` regenerate figures.

---

## 🧠 Key Concepts

| Term | Meaning |
|---|---|
| **D0** | Undefended base model, neutral prompt |
| **D1** | Undefended base model + explicit anti-injection prompting defense |
| **D2** | DPO-trained adapter, queried with the neutral prompt |
| **D3** | SFT-only ablation adapter |
| **F1-F5** | Attack families (F1-F3 used in training, F4-F5 held out) |
| **ASR** | Attack Success Rate |
| **FDR** | Flaw-Detection Recall |
| **DID** | Paired difference-in-differences protocol |

---

## ⚠️ Known Behaviors

- **Frontier-backend sampling.** Explicitly set `do_sample` to ensure determinism across API and local backends.
- **D3+F2 malformed outputs.** A small number of D3 responses on the F2 attack family may be unparseable at the 150-token budget.

---

## 📊 Data and Model Weights

Raw PDFs, extracted manuscript text, trained LoRA adapters, and base model weights are excluded (too large). Run `download_arxiv_corpus.py` and `download_models.py` to regenerate them. `data/processed/revguard_bench/` and `data/processed/preference_pairs/` are the core datasets.

---

## 📝 License
MIT License. See the [LICENSE](LICENSE) file for details.
