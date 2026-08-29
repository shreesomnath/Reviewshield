<div align="center">
  
  # Reviewshield 🛡️
  
  **Defending LLM Reviewers Against In-Paper Prompt Injection**

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org/)
  [![Framework](https://img.shields.io/badge/Framework-PyTorch%202.7-orange)](#)
  
  <br />
</div>

> **Reviewshield** is a comprehensive code and benchmark suite designed to defend LLM reviewers against in-paper prompt injections under instruction-content entanglement.

---

## 🎯 The Challenge: Instruction-Content Entanglement

An author who controls their own manuscript can embed hidden instructions engineered to manipulate whichever LLM reviews it. Defending against this is vastly harder than generic prompt injection because of **instruction-content entanglement**:

- 🚧 **Untrusted Channel:** The manuscript itself carries the attack.
- ⚖️ **Legitimate Engagement:** The reviewer must still engage critically with the document.
- 🎯 **Dual Requirement:** A defense cannot just learn to ignore instruction-like text without simultaneously breaking the review process itself.

---

## ✨ Key Features & Repository Layout

This repository contains **ReviewShield-Bench**, a benchmark of five in-paper injection families embedded in real open-access manuscripts, alongside a complete training and evaluation pipeline.

### 📊 Corpus & Benchmark Assembly
Scripts to construct the dataset from arXiv papers, run content-quality audits, and assemble ReviewShield-Bench (spanning clean, injection-only, injection+flaw, and flaw-only variants).

### 🧠 Training Pipeline
Robust training methodologies utilizing a DPO-trained defense, an SFT-only ablation, and shared neutral reviewer prompts to adapt base models into resilient reviewers.

### 🔬 Comprehensive Evaluation Harness
End-to-end evaluation focusing on Attack Success Rate (ASR) metrics, Flaw-Detection Recall (FDR), and over-refusal rates. 

### 📈 Difference-in-Differences (DID) Protocol
A paired statistical protocol introduced to correctly measure a defense's effect once its own scoring calibration is accounted for, eliminating confounding variables in evaluation.

---

## 🏗️ Repository Structure

```text
Reviewshield/
├── scripts/
│   ├── download_arxiv_corpus.py   Pull open-access CS/ML papers from arXiv
│   ├── benchmark_assembly.py      Assemble ReviewShield-Bench
│   ├── attacks/                   F1-F5 attack families
│   ├── generation/                Model loading, API clients, dataset generation
│   ├── training/                  DPO and SFT training scripts
│   ├── evaluation/                Main harness and evaluation scripts
│   └── analysis/                  DID protocol, statistical analysis, plotting
├── environment/
│   └── revguard.def               Apptainer/Singularity container definition
└── run/                           Shell scripts to run pipeline stages
```

---

## 🚀 Environment & Setup

Everything runs inside the Apptainer container defined in `environment/revguard.def` (PyTorch 2.7.0 + CUDA 12.8, Python 3.11).

1. **Build the container:**
   ```bash
   apptainer build --fakeroot --nv revguard.sif environment/revguard.def
   ```

2. **Execute a script:**
   ```bash
   apptainer exec --nv --bind /path/to/project:/workspace revguard.sif \
       python /workspace/scripts/<script>.py
   ```

*(Credentials are passed securely at runtime via environment variables).*

---

## 📊 Core Concepts

| Term | Meaning |
|---|---|
| **D0 / D1** | Undefended base model / Undefended base model + explicit prompting defense |
| **D2 / D3** | DPO-trained adapter / SFT-only ablation adapter |
| **F1-F5** | Attack families (Direct override, stealth, impersonation, authority, sycophancy) |
| **ASR** | Attack Success Rate (score-inflation or decision-flip) |
| **FDR** | Flaw-Detection Recall |

---

## 👨‍💻 Author

Proudly engineered and developed by:

* **Somnath Luitel**

<div align="center">
  <br/>
  <i>Defending the integrity of LLM-assisted reviews.</i>
</div>
 
 
 
 
 
 
