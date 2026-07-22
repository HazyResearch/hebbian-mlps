# Qwen3 Activation Artifacts

The Qwen3 experiments use paired inputs and outputs from layer 14 of
`Qwen/Qwen3-0.6B-Base`. These tensors are too large for git and must be
generated before running the Qwen margin and capacity experiments.

Install the optional dependencies and capture the paper-scale bundle:

```bash
python -m pip install -e '.[llm]'
QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14 \
  python scripts/paper/run.py prepare_qwen3_layer14_artifacts
```

The resulting bundle has this layout:

```text
qwen3_06b_layer14/
  activations/
    x.pt
    y.pt
    metadata.json
  results/
```

`x.pt` and `y.pt` are paired tensors with shape `[N, 1024]`. The paper bundle
uses 500,000 rows: `x` is the layer's post-attention normalization output and
`y` is its MLP output before the residual addition.

Validate the activation bundle with:

```bash
QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14 \
QWEN3_CHECK_SCOPE=activations \
QWEN3_EXPECTED_D_MODEL=1024 \
QWEN3_MIN_ROWS=500000 \
  python scripts/paper/run.py check_qwen3_artifacts
```

The [paper experiment guide](../../../../PAPER_EXPERIMENT_MAP.md) lists the
three Qwen entry points that consume this bundle.
