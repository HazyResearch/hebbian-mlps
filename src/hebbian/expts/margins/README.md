# Margin Sweeps

Scripts to replicate random-feature, anisotropic-beta, Qwen3-margin, and
noisy-query margin paper results.

Paper targets:

```bash
python scripts/paper/run.py fig_rf_margin_m --mode run-and-plot
python scripts/paper/run.py fig_rf_margin_f --mode run-and-plot
python scripts/paper/run.py fig_beta_margin_sweeps --mode run-and-plot
python scripts/paper/run.py fig_llm_margin_qwen3_layer14 --mode run-and-plot
python scripts/paper/run.py fig_noisy_query_margin --mode run-and-plot
```

Python modules:

```bash
python -m hebbian.expts.margins.run
python -m hebbian.expts.margins.run_qwen3
python -m hebbian.expts.margins.plot
python -m hebbian.expts.margins.plot_noisy_query
```

`run.py` handles synthetic and non-LLM sweeps. `run_qwen3.py` produces the
Qwen3 AK/AV result and its matched RK/RV synthetic control.
`theory.py` contains the fitted theorem forms used by the canonical plotter.
