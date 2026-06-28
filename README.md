# Text-to-SQL Fine-Tuning with QLoRA

Fine-tuning an open-weight LLM (Phi-3-mini / Llama-3.1-8B) with QLoRA for natural-language-to-SQL generation, with full experiment tracking, rigorous evaluation against baselines, and production-style serving.

> **Status:** 🚧 In progress — use this README as the build plan / checklist. Fill in results as you complete each phase.

---

## Headline Result

> _Fill in once Phase 4 is complete:_
> Fine-tuned **[model name]** using QLoRA on the Spider text-to-SQL dataset, improving execution accuracy from **X%** (few-shot baseline) to **Y%**; served via **[vLLM / llama.cpp]** + FastAPI with **Nms** p95 latency, containerized with Docker.

---

## Table of Contents

- [Why This Project](#why-this-project)
- [Architecture Overview](#architecture-overview)
- [Phase 0 — Environment Setup](#phase-0--environment-setup)
- [Phase 1 — Data](#phase-1--data)
- [Phase 2 — Baseline](#phase-2--baseline-do-this-before-fine-tuning)
- [Phase 3 — Training](#phase-3--training)
- [Phase 4 — Evaluation](#phase-4--evaluation)
- [Phase 5 — Production Packaging](#phase-5--production-packaging)
- [Phase 6 — Repo Polish](#phase-6--repo-polish)
- [Results](#results)
- [Error Analysis](#error-analysis)
- [Reproduce](#how-to-reproduce)
- [Future Work](#future-work)

---

## Why This Project

Most "GenAI" portfolio projects call a hosted API (OpenAI, Anthropic, etc.) and add a thin wrapper — there's no training involved, and no evidence the candidate understands model internals. This project is built specifically to demonstrate:

- Actual fine-tuning (QLoRA), not prompting
- A baseline comparison that proves fine-tuning helped, with real numbers
- Production-style serving and evaluation discipline, not just a notebook

Text-to-SQL is the chosen task because it has **objective, automatically-checkable evaluation** (does the generated query execute and return the correct result?) and a standard benchmark (Spider) that hiring managers and technical interviewers will recognize.

---

## Architecture Overview

```
Raw Spider dataset
        │
        ▼
  Data conversion script (schema + question → instruction format)
        │
        ▼
   Train / Val / Test split (test = untouched Spider dev set)
        │
        ├──────────────┐
        ▼              ▼
  Baseline eval    QLoRA fine-tuning (PEFT + bitsandbytes + TRL SFTTrainer)
  (few-shot &          │
   zero-shot,          ▼
   no training)    W&B experiment tracking (loss, LR sweep, configs)
        │              │
        └──────┬───────┘
               ▼
        Evaluation (exact match + execution accuracy)
               │
               ▼
        Error analysis (categorized failure modes)
               │
               ▼
   Merge adapters → quantize/export (GGUF) → serve via vLLM/llama.cpp
               │
               ▼
        FastAPI wrapper → Docker container → load test (latency, RPS)
```

---

## Phase 0 — Environment Setup

**Model choice:**
- Start with **Phi-3-mini (3.8B)** — fast to iterate, fits comfortably with QLoRA on a single 16GB GPU (free-tier Colab/Kaggle T4 works).
- Once the pipeline is validated end-to-end, scale up to **Llama-3.1-8B** for the "real" results.

**Compute options:**
- Kaggle (free T4, 16GB, 30 hrs/week)
- Google Colab Pro
- Rented GPU (RunPod / Lambda Labs) — a few dollars for an A10 if you need more headroom

**Dependencies:**
```bash
pip install torch transformers peft bitsandbytes trl accelerate datasets evaluate wandb
```

**Experiment tracking:**
- [ ] Create a Weights & Biases account (free tier)
- [ ] `wandb login` and confirm a test run logs correctly
- [ ] Log **every** run from the start — don't try to reconstruct logs retroactively

---

## Phase 1 — Data

**Dataset:** [Spider](https://yale-lily.github.io/spider) — ~10K examples, multi-domain, multi-table. (Harder and more credible than WikiSQL since it's the standard academic benchmark — your numbers become comparable to published results.)

**Target instruction format:**
```json
{
  "instruction": "Given the database schema, write a SQL query to answer the question.",
  "schema": "CREATE TABLE singer (Singer_ID, Name, Country, Age) ...",
  "question": "How many singers are from France?",
  "output": "SELECT COUNT(*) FROM singer WHERE Country = 'France'"
}
```

**Checklist:**
- [ ] Write `scripts/convert_spider_to_instruction_format.py` — converts raw Spider JSON + schema files into the format above
- [ ] Split: train/val from Spider's `train` set
- [ ] **Test set = Spider's `dev` set — do not touch until Phase 4**
- [ ] Manually inspect ~50 converted examples — note schema-linking issues, ambiguous questions, edge cases
- [ ] Write up inspection notes (this feeds directly into your error analysis section later)

---

## Phase 2 — Baseline (do this *before* fine-tuning)

This is the most important step in the whole project — skip it and you have no way to prove fine-tuning actually helped.

**Checklist:**
- [ ] Run **base model + zero-shot prompting** on the test set
- [ ] Run **base model + few-shot prompting** (3–5 in-context examples) on the test set
- [ ] Compute exact-match and execution accuracy for both
- [ ] Record results in the [Results](#results) table below

---

## Phase 3 — Training

**Method:** QLoRA (4-bit NF4 quantized base model + LoRA adapters) via `peft` + TRL's `SFTTrainer`.

**Config decisions to make deliberately — and log why:**

| Hyperparameter | Starting value | Sweep range |
|---|---|---|
| LoRA rank (r) | 16 | 8, 16, 32 |
| LoRA target modules | q/k/v projections | — |
| Learning rate | 2e-4 | 1e-4, 2e-4, 5e-4 |
| Epochs | 3 | watch val loss for overfitting |
| Quantization | 4-bit NF4 | — |

**Checklist:**
- [ ] Implement training script using `SFTTrainer`
- [ ] Run **at least 3 configs** (vary rank, LR, or target modules) — log all in W&B
- [ ] Monitor for catastrophic forgetting — spot-check general instruction-following ability after fine-tuning
- [ ] Monitor train/val loss curves for overfitting to Spider's schema patterns
- [ ] Save best checkpoint(s) + W&B run links

---

## Phase 4 — Evaluation

**Metrics (both required):**
- **Exact match accuracy** — normalized string match against gold SQL
- **Execution accuracy** — generated SQL run against the actual database, compared to gold query's result set (this is the metric that matters most — two queries can differ textually but be logically equivalent)

**Required comparison table:**

| Model | Exact Match | Execution Accuracy |
|---|---|---|
| Base model, zero-shot | | |
| Base model, few-shot | | |
| **Fine-tuned (QLoRA)** | | |

- [ ] Run fine-tuned model on the held-out Spider dev set
- [ ] Populate the table above
- [ ] Write 2–3 sentences interpreting the gap between baseline and fine-tuned results

---

## Error Analysis

Categorize failure cases on the test set. Suggested buckets:

- [ ] Wrong table joins
- [ ] Wrong aggregation function (COUNT vs SUM vs AVG)
- [ ] Schema misunderstanding (wrong column/table referenced)
- [ ] Syntactically invalid SQL
- [ ] Semantically equivalent but textually different (false negative on exact match)

For each bucket: count of occurrences, 1–2 representative examples, and a brief note on likely cause.

---

## Phase 5 — Production Packaging

**Checklist:**
- [ ] Merge LoRA adapters into base model weights
- [ ] Export to GGUF (via llama.cpp conversion scripts) for CPU-friendly quantized inference, **or** keep adapters + base separate for GPU serving
- [ ] Stand up serving:
  - GPU path: **vLLM**
  - CPU path: **llama.cpp server**
- [ ] Wrap with **FastAPI** — clean `/generate` endpoint, input validation on schema format
- [ ] Add latency logging (p50/p95) per request
- [ ] Write a `Dockerfile` and confirm the container runs standalone
- [ ] Load test with `locust` (or a simple async script) — report requests/sec and p50/p95 latency

---

## Phase 6 — Repo Polish

Final README should read, top to bottom:

1. One-line problem statement + headline result
2. Architecture diagram (this file already has one — update with real numbers)
3. Results table (filled in)
4. Reproduction instructions (commands for setup → train → eval → serve)
5. Error analysis section
6. "What I'd do with more time/compute" — signals seniority of thinking

---

## How to Reproduce

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Convert data
python scripts/convert_spider_to_instruction_format.py \
  --spider_dir data/spider \
  --output_dir data/processed

# 3. Run baseline (no training)
python scripts/eval_baseline.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --mode few-shot \
  --test_set data/processed/dev.json

# 4. Train
python scripts/train_qlora.py \
  --base_model microsoft/Phi-3-mini-4k-instruct \
  --train_data data/processed/train.json \
  --val_data data/processed/val.json \
  --lora_r 16 \
  --lr 2e-4 \
  --epochs 3 \
  --wandb_project text2sql-qlora

# 5. Evaluate fine-tuned model
python scripts/eval_finetuned.py \
  --adapter_path outputs/checkpoint-best \
  --test_set data/processed/dev.json

# 6. Merge + export
python scripts/merge_and_export.py \
  --base_model microsoft/Phi-3-mini-4k-instruct \
  --adapter_path outputs/checkpoint-best \
  --output_path outputs/merged-gguf

# 7. Serve
docker build -t text2sql-api .
docker run -p 8000:8000 text2sql-api

# 8. Load test
locust -f scripts/loadtest.py --host http://localhost:8000
```

---

## Future Work

_Fill in after completing the project — e.g., scaling to Llama-3-8B, multi-table reasoning improvements, RAG-based schema retrieval for large databases, RLHF/DPO pass on top of SFT, etc._

---

## Resume Bullet (once complete)

> Fine-tuned Phi-3-mini (3.8B) using QLoRA on the Spider text-to-SQL benchmark, improving execution accuracy from X% (few-shot baseline) to Y%; served via vLLM/FastAPI with sub-Nms p95 latency, containerized with Docker.
