# Optimizer Comparison for GPT Language Model Training

An EPFL OptML course project investigating how optimizer choice affects both training performance and representational quality of a small GPT language model.

## Overview

We train a ~50M parameter GPT model from scratch on educational web text and compare four optimizers — **AdamW**, **Adam**, **Muon**, and **SGD** — on two axes:

1. **Training efficiency**: cross-entropy loss and perplexity over time
2. **Representation quality**: [RankMe](https://arxiv.org/abs/2210.02885) score, which measures the effective rank of the model's last-layer features via SVD entropy. Higher RankMe indicates more diverse, expressive representations.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `LM_train.ipynb` | Train the GPT model with any of the four optimizers, compare loss/perplexity curves, and save checkpoints |
| `rankme_analysis.ipynb` | Load saved checkpoints and compute RankMe scores across training steps for each optimizer |

## Model Architecture

A small but modern GPT variant (~50M params, 8 layers, 512 hidden dim) with:
- Rotary Position Embeddings (RoPE)
- RMSNorm + QK-Norm for training stability
- Value Embeddings (ResFormer-style residual shortcuts)
- Per-layer residual scaling (`resid_lambdas`, `x0_lambdas`)
- Squared ReLU feed-forward network
- Logit soft-capping

## Optimizers

| Optimizer | Notes |
|-----------|-------|
| **AdamW** | Adam with decoupled weight decay; peak LR 3e-4 |
| **Adam** | No weight decay; peak LR 3e-4 |
| **Muon** | Second-order-inspired; applies to hidden 2D weights only at LR 0.02; embeddings/head use AdamW |
| **SGD** | Nesterov momentum; peak LR 3e-4 |

All runs use a warmup → constant → cooldown LR schedule on the DCLM-edu dataset with a 5400s time budget.

## RankMe

RankMe quantifies how well a model uses its representational capacity:

$$\text{RankMe} = \exp\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\lambda_i}{\|\boldsymbol{\lambda}\|_1} + \epsilon$$

where $\boldsymbol{\lambda}$ are the eigenvalues of the batch of last-layer feature vectors. A higher score means the model distributes information across more dimensions rather than collapsing into a low-rank subspace.

## Setup

```bash
bash install.sh
```

A CUDA GPU is required for reasonable training speed, and PyTorch >= 2.9 is required for the Muon optimizer implementation to be available in torch.optim. The notebooks are self-contained and clone [karpathy/autoresearch](https://github.com/karpathy/autoresearch) automatically.

## Usage

1. Run `LM_train.ipynb` — select dataset, optimizer, and time budget in Sections 1–2, then run all cells. Checkpoints are saved to `./checkpoints/<optimizer>/`.
2. Run `rankme_analysis.ipynb` after training to compare RankMe trajectories across optimizers.
