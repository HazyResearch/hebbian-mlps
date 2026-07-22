# Hidden-Dim Sweep

Scripts to replicate the hidden-dimension paper result family.

Paper targets:

```bash
python scripts/paper/run.py fig_hidden_dim_margin --mode run-and-plot
python scripts/paper/run.py fig_margin_violin --mode run-and-plot
```

Python modules:

```bash
python -m hebbian.expts.hidden_dim.run
python -m hebbian.expts.hidden_dim.plot_dual_axis
python -m hebbian.expts.hidden_dim.plot_hidden_dim_sweep
python -m hebbian.expts.hidden_dim.plot_margin_distribution
```
