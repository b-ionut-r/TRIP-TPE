# TRIP-TPE

**Trajectory-Informed Region Proposal + Tree-structured Parzen Estimator**

A hybrid meta-learning framework that augments Optuna's TPE with a lightweight Transformer for learned hyperparameter search space region proposals. The Transformer handles macro-region definition (coarse guidance), while TPE retains full authority over fine-grained density estimation and sampling.

## Key Features

- **Drop-in Optuna sampler** — implements `BaseSampler`, works with any Optuna study
- **Three-phase sampling** — warmup → Transformer region proposal → constrained TPE
- **Meta-feature conditioning** — 10-dimensional dataset prior for cold-start acceleration
- **Adaptive trust scheduling** — self-correcting feedback loop modulates Transformer influence
- **DETR-style dimension queries** — per-dimension cross-attention (no [CLS] bottleneck)
- **Mixture-of-Betas output** — K=3 component mixture captures multimodal optima
- **FP16 mixed-precision training** — fits on a single 4 GB GPU
- **W&B integration** — live logging for data generation, training, and benchmarks
- **Graceful degradation** — falls back to pure TPE if model is unavailable

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  TRIPTPESampler                   │
│                (BaseSampler)                      │
│                                                  │
│  Phase 1: Warmup (random/TPE)                    │
│  Phase 2: Transformer → region bounds [0,1]^D    │
│  Phase 3: apply_region_bounds → constrained TPE  │
│                                                  │
│  Trust factor: 0.8 → 0.3 (adaptive decay)        │
│  Requery interval: every 10 trials               │
└──────────────────────────────────────────────────┘
```

## Quick Start

```python
import optuna
from trip_tpe import TRIPTPESampler

sampler = TRIPTPESampler(
    model_path="checkpoints/best_model.pt",
    n_warmup_trials=5,
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
```

## Installation

```bash
pip install -e .
```

## Full Training Pipeline

### 0. Prepare benchmark data

```bash
# HPO-B: the Python handler does not download the dataset for you.
# Download hpob-data.zip and unpack it under data/hpob/.
# Official source: https://github.com/machinelearningnuremberg/HPO-B

# YAHPO Gym: download the surrogate bundle and point yahpo_gym at it.
# Official setup docs: https://slds-lmu.github.io/yahpo_gym/getting_started.html
```

### 1. Generate training data

```bash
# Synthetic (for local testing)
trip-tpe-generate --mode synthetic --n-trajectories 50000 --output data/train.pt

# Real surrogate mix (recommended for production)
trip-tpe-generate --mode real --output data/real_train.pt
```

### 2. Train the Transformer

```bash
trip-tpe-train --data-path data/real_train.pt --mix-synthetic 0.1 --epochs 100
```

### 3. Benchmark

```bash
trip-tpe-eval --model-path checkpoints/best_model.pt --benchmarks yahpo hpob synthetic --training-manifest data/training_manifest.json --n-seeds 20
```

All three stages support `--wandb-project` and `--wandb-entity` flags for live W&B logging.

## Project Structure

```
trip_tpe/
├── models/          # Transformer architecture (~10M params)
├── samplers/        # Optuna BaseSampler integration
├── data/            # Preprocessing, datasets, trajectory generation
├── training/        # Training loop, loss functions
├── evaluation/      # Benchmarking framework
└── utils/           # Config, search space encoding, metrics
configs/             # YAML configuration files
tests/               # Unit tests
```

## Configuration

All hyperparameters are in `trip_tpe/utils/config.py` with YAML serialization:

```bash
trip-tpe-train --config configs/default.yaml
```

## Citation

If you use TRIP-TPE in your research, please cite:

```bibtex
@software{trip_tpe_2026,
  title={TRIP-TPE: Trajectory-Informed Region Proposal for Tree-structured Parzen Estimation},
  author={Ionuț},
  year={2026},
  url={https://github.com/b-ionut-r/TRIP-TPE}
}
```

## License

MIT
