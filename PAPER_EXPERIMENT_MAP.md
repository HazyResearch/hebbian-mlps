# Paper Experiments

The paper command line is [`scripts/paper/run.py`](scripts/paper/run.py). It
keeps experiment names visible while the implementation is grouped into small
modules by experiment family:

```bash
python scripts/paper/run.py --list
python scripts/paper/run.py <target> --mode run-and-plot
python scripts/paper/run.py <target> --mode plot-only
```

Results default to `artifacts/paper/results/` and figures to
`artifacts/paper/figures/`. Common paths and GPU settings are command line
options; run `python scripts/paper/run.py --help` for the full list.

## Margin Experiments

These test the margin predictions behind the Hebbian construction, including
the Figure 2 hidden-dimension and random-feature studies and appendix sweeps
over fact count, anisotropy, query noise, and per-key margins.

```bash
python scripts/paper/run.py fig_hidden_dim_margin --mode run-and-plot
python scripts/paper/run.py fig_rf_margin_m --mode run-and-plot
python scripts/paper/run.py fig_rf_margin_f --mode run-and-plot
python scripts/paper/run.py fig_beta_margin_sweeps --mode run-and-plot
python scripts/paper/run.py fig_noisy_query_margin --mode run-and-plot
python scripts/paper/run.py fig_margin_violin --mode run-and-plot
```

Paper orchestration is in
[`targets/margins.py`](scripts/paper/targets/margins.py) and
[`targets/capacity.py`](scripts/paper/targets/capacity.py). Experiment and
plotting implementations live under
[`src/hebbian/expts/margins/`](src/hebbian/expts/margins/) and
[`src/hebbian/expts/hidden_dim/`](src/hebbian/expts/hidden_dim/).

## Standalone MLP Capacity

These compare trained MLPs, the three paper Hebbian constructions, and the NTK
baseline. They produce the isotropic capacity panel in Figure 2 and the
anisotropic appendix result.

```bash
python scripts/paper/run.py fig_mlp_capacity_isotropic --mode run-and-plot
python scripts/paper/run.py fig_mlp_capacity_anisotropic --mode run-and-plot
```

Paper orchestration is in
[`targets/capacity.py`](scripts/paper/targets/capacity.py); the runner and
plotter live in
[`src/hebbian/expts/mlp_capacity/`](src/hebbian/expts/mlp_capacity/).

## Transformer Capacity

These measure the attention noise floor and the number of facts a Transformer
block can store. They produce Figure 3 panels and the appendix result at a 99%
train-accuracy threshold.

```bash
python scripts/paper/run.py fig_attention_noise_floor --mode run-and-plot
python scripts/paper/run.py fig_transformer_storage_capacity --mode run-and-plot
python scripts/paper/run.py fig_transformer_capacity_train99 --mode run-and-plot
```

Paper orchestration is in
[`targets/capacity.py`](scripts/paper/targets/capacity.py). The experiment code
is under
[`src/hebbian/expts/attention_noise/`](src/hebbian/expts/attention_noise/) and
[`src/hebbian/expts/transformer/`](src/hebbian/expts/transformer/).

## Qwen3 Activation Experiments

These repeat the margin, standalone MLP-capacity, and Transformer-capacity
studies using layer-14 activations from `Qwen/Qwen3-0.6B-Base`.

```bash
python scripts/paper/run.py prepare_qwen3_layer14_artifacts
python scripts/paper/run.py check_qwen3_artifacts
python scripts/paper/run.py fig_llm_margin_qwen3_layer14 --mode run-and-plot
python scripts/paper/run.py fig_llm_mlp_capacity_qwen3_layer14 --mode run-and-plot
python scripts/paper/run.py fig_llm_transformer_capacity_qwen3_layer14 --mode run-and-plot
```

Paper orchestration and artifact checks are in
[`targets/qwen.py`](scripts/paper/targets/qwen.py). See the
[Qwen3 artifact guide](src/hebbian/expts/llm_embeddings/README.md) for the
activation format and capture workflow.

## Fact Editing

This experiment trains a small Transformer, replaces or edits its MLP fact
store, and evaluates editing quality and non-fact perplexity.

```bash
python scripts/paper/run.py fig_fact_editing_score --mode run-and-plot
python scripts/paper/run.py fig_fact_editing_nonfact_ppl_ratio --mode plot-only
python scripts/paper/run.py tab_fact_editing_h512 --mode plot-only
```

Paper orchestration is in
[`targets/fact_editing.py`](scripts/paper/targets/fact_editing.py), and the
multi-GPU paper grid is implemented by
[`pipeline.py`](src/hebbian/expts/fact_editing/pipeline.py).

## Main Figure Assembly

After generating the component panels, use
[`plot_fig2.py`](scripts/paper/plot_fig2.py) or
[`plot_fig3.py`](scripts/paper/plot_fig3.py) to assemble the corresponding
vector figure.
