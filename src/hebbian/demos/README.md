# Demos

This folder contains small, runnable demos for the MLP variants in this repo. Each script builds a synthetic `Factset`, constructs or trains an MLP, and reports accuracy (plus a few metrics).

## Quick Start

All demos are runnable as modules from the repo root:

```bash
python -m hebbian.demos.gd_demo
python -m hebbian.demos.ntk_demo
python -m hebbian.demos.hebbian_demo
```

Paper and comparison sweeps live under `src/hebbian/expts/`.

## Demos

`gd_demo.py`
- Gradient‑descent trained MLP on a synthetic factset.
- Key args: `--d-model`, `--facts-multiplier`, `--num-epochs`, `--activation`.

`ntk_demo.py`
- NTK construction on a synthetic factset.
- Key args: `--m`, `--hermite-degree`, `--activation`.

`hebbian_demo.py`
- Paper Hebbian MLP variants (`unwhitened`, `whitened`, `data_dependent`).
- Key args: `--variant`, `--m`.

## Example Commands

```bash
# GD MLP
python -m hebbian.demos.gd_demo --d-model 64 --facts-multiplier 0.25

# NTK MLP
python -m hebbian.demos.ntk_demo --m 1000 --activation swish

# Hebbian variant
python -m hebbian.demos.hebbian_demo --variant data_dependent

```
