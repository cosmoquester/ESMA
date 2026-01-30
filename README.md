# ESMA: Fine-Tuning Language Models to Know What They Know

This repository contains the code for **Evolution Strategy for Metacognitive Alignment (ESMA)**, a method to improve large language models’ awareness of their own knowledge. The work is described in:

## Overview

Metacognition, knowing what one knows, is central to intelligence. This project provides:

1. **A measurement framework** for LLM metacognition using a dual-prompt protocol (Direct Questions + Meta Questions) and the **d′<sub>type2</sub>** metric from signal detection theory.
2. **ESMA**: evolution-strategy-based fine-tuning that strengthens the link between a model’s internal knowledge and its explicit answers, including "Do you know the answer?" style meta-questions.
3. **Evaluation tools** for d′<sub>type2</sub>, raw alignment, accuracy, yes/no ratios, and related metrics on TriviaQA and other QA datasets.

ESMA uses a population of weight-perturbed models, a joint reward over direct correctness and meta-alignment, and weighted averaging of parameters. It improves metacognitive sensitivity (e.g. d′<sub>type2</sub> ≈ 1) and generalizes to unseen prompts, languages, and datasets.

## Installation

```bash
pip install -e .
```

## Repository Structure

- **`esma/`** – Core library:
  - **`metric.py`** – d′<sub>type2</sub>, raw alignment, yes/no detection, RMI.
  - **`reward.py`** – ESMA joint reward and ablations (correctness-only, alignment-only).
  - **`evolution.py`** – Evolution strategy (perturbation, evaluation, weighted update).
  - **`prompt.py`** – Direct / Meta / IDK prompt templates.
  - **`dataset.py`** – Dataset and data loading utilities.
  - **`data/`** – TriviaQA, FreebaseQA, NQ Open, WebQuestions, MKQA, FictionalQA, etc.
- **`scripts/`**
  - **`train_es.py`** – ESMA training (evolution strategy on TriviaQA).
  - **`train_sft_meta.py`** – Supervised fine-tuning for meta-answers (SFT baseline).
  - **`train_sft.py`** – General SFT (e.g. for FictionalQA).
  - **`evaluate_qa.py`** – Evaluate models on dual-prompt QA (d′<sub>type2</sub>, alignment, accuracy).
  - **`evaluate_qa_idw.py`** – “I don’t know” (IDK) single-prompt evaluation.
  - **`evaluate_qa_threshold.py`** – Threshold-based / confidence evaluation.
  - **`evaluate_qa_api.py`** – Evaluation for API-based models.
  - **`apply_weight_change.py`** – Apply (e.g. sparse) weight deltas to a base model.
  - **`extract_weight_change.py`** – Extract weight changes (e.g. for patching analysis).
- **`notebooks/`** – Plotting confidence distributions and weight-patching effects.

## Quick Start

### Train with ESMA (evolution strategy)

```bash
python scripts/train_es.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --sigma 1e-3 --alpha 5e-4 \
  --num-iterations 750 --population-size 32 \
  --num-data-per-iteration 256 \
  --reward-type esma
```

Hyperparameters (e.g. σ, α, iterations, population size) follow the paper; `--reward-type esma` uses the joint reward (correctness + meta-alignment).

### Evaluate a model (dual-prompt metacognition)

```bash
python scripts/evaluate_qa.py --model path/to/model
```

Use `--help` for data paths, batch size, and output options.

## Citation

If you use this code or the method, please cite:

```bibtex
TBD
```
