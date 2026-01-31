# ESMA: Fine-Tuning Language Models to Know What They Know

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/cosmoquester/ESMA/tree/main.svg?style=svg&circle-token=CCIPRJ_3Rem2YqPXQh2HkCjUBcccb_6ae74c3a5b9fffc96541ddaa19bc0b9c018ed0fa)](https://dl.circleci.com/status-badge/redirect/gh/cosmoquester/ESMA/tree/main)
[![codecov](https://codecov.io/gh/cosmoquester/ESMA/graph/badge.svg?token=vG6ctZxQuk)](https://codecov.io/gh/cosmoquester/ESMA)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org)

This repository contains the code for **Evolution Strategy for Metacognitive Alignment (ESMA)**, a method to improve large language models’ awareness of their own knowledge. The work is described in:

## Overview

Metacognition, knowing what one knows, is central to intelligence. This project provides:

1. **A measurement framework and evaluation tools** for LLM metacognition: a dual-prompt protocol (Direct Questions + Meta Questions), the **d′<sub>type2</sub>** metric from signal detection theory, and evaluation for d′<sub>type2</sub>, raw alignment, accuracy, yes/no ratios, and related metrics on TriviaQA and other QA datasets.
2. **ESMA**: evolution-strategy-based fine-tuning that strengthens the link between a model’s internal knowledge and its explicit answers, including "Do you know the answer?" style meta-questions.
3. **Evolution strategy weight patching scripts** to extract weight deltas (tuned − base) and apply sparse or full updates (e.g. top/bottom *p*% by magnitude) for analyzing which parameter changes drive metacognitive improvement ([paper](https://nn.cs.utexas.edu/downloads/papers/park.metacognition-0126.pdf), §5.6).

ESMA uses a population of weight-perturbed models, a joint reward over direct correctness and meta-alignment, and weighted averaging of parameters. It improves metacognitive sensitivity (e.g. d′<sub>type2</sub> ≈ 1) and generalizes to unseen prompts, languages, and datasets.

## Installation

```bash
git clone https://github.com/cosmoquester/ESMA.git && cd ESMA
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

### Train with ESMA

```bash
python scripts/train_es.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --sigma 1e-3 --alpha 5e-4 \
  --num-iterations 750 --population-size 32 \
  --num-data-per-iteration 256 \
  --reward-type esma
```

Hyperparameters (e.g. σ, α, iterations, population size) follow the paper; `--reward-type esma` uses the joint reward (correctness + meta-alignment).

### Evaluate a model

```bash
python scripts/evaluate_qa.py --model path/to/model
```

Use `--help` for data paths, batch size, and output options.

## Citation

If you use this code or the method, please cite:

```bibtex
TBD
```
