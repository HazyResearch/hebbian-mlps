# Paper Commands

`run.py` is the public entry point for paper experiments and plots:

```bash
python scripts/paper/run.py --list
python scripts/paper/run.py fig_rf_margin_m --mode run-and-plot
python scripts/paper/run.py fig_rf_margin_m --mode plot-only
```

The code is organized by responsibility:

```text
scripts/paper/
|-- run.py                 # Small CLI and explicit target registry
|-- common.py              # Paths, subprocess execution, and asset copying
|-- targets/
|   |-- margins.py         # Synthetic margin sweeps and plots
|   |-- capacity.py        # Hidden dim, MLP, attention, and Transformer capacity
|   |-- fact_editing.py    # Fact-editing plots and table
|   `-- qwen.py            # Qwen capture, validation, and experiments
|-- plot_fig2.py           # Assemble the Figure 2 vector composite
`-- plot_fig3.py           # Assemble the Figure 3 vector composite
```

Common settings are command line flags:

```bash
python scripts/paper/run.py fig_mlp_capacity_isotropic \
  --mode run-and-plot \
  --n-gpus 2 \
  --result-root /path/to/results \
  --figure-root /path/to/figures
```

Target-specific research overrides retain their environment variable names:

```bash
RF_MARGIN_M_N_POINTS=10 \
  python scripts/paper/run.py fig_rf_margin_m --mode run-and-plot
```

Generated assets can optionally be copied into a paper checkout with
`--paper-root /path/to/mlps-are-hebbians-paper`.
