# MLP Capacity

Scripts to replicate standalone MLP capacity sweeps and plots.

Paper targets:

```bash
python scripts/paper/run.py fig_mlp_capacity_isotropic --mode run-and-plot
python scripts/paper/run.py fig_mlp_capacity_anisotropic --mode run-and-plot
python scripts/paper/run.py fig_llm_mlp_capacity_qwen3_layer14 --mode run-and-plot
```

Python modules:

```bash
python -m hebbian.expts.mlp_capacity.run
python -m hebbian.expts.mlp_capacity.run_qwen3
python -m hebbian.expts.mlp_capacity.plot
```

`run.py` handles the isotropic and anisotropic sweeps. `run_qwen3.py` runs the
same binary-search experiment with Qwen3 activation embeddings.
