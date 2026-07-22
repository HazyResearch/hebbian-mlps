# Experiment Artifacts

Paper-scale outputs are generated locally and are not tracked in git. The
paper command line uses this layout by default:

```text
artifacts/paper/
  results/   experiment outputs consumed by plotters
  figures/   generated PNG, PDF, CSV, and TeX figure products
```

Use `--result-root`, `--figure-root`, or `--artifact-root` when storing outputs
elsewhere. Plot-only mode reads the default result tree unless a target's input
environment variable is set; see
[`PAPER_EXPERIMENT_MAP.md`](../PAPER_EXPERIMENT_MAP.md).

## Qwen3 Layer-14 Bundle

The Qwen3 experiments require activation pairs that are too large for git. Use
one external root for activations and all derived results:

```bash
export QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14
```

Expected layout:

```text
$QWEN3_ARTIFACT_ROOT/
  activations/
    x.pt
    y.pt
    metadata.json
  results/
    margins/
      qwen3_06b_layer14_F_akav.json
      qwen3_06b_layer14_F_rkrv.json
    mlp_capacity/
      <method>/d<d>_F<num_facts>/.../binary_search_results_*.pkl
    transformer_capacity/
      capacity_points.csv
```

`x.pt` and `y.pt` are matching 2D tensors. For the paper experiment, `x.pt` is
the layer-14 `post_attention_layernorm` output and `y.pt` is the layer's MLP
output before the residual addition. Layer 14 is the zero-based Hugging Face
decoder-layer index for `Qwen/Qwen3-0.6B-Base`.

## Generate Activations

Install the optional dependencies, then run the public capture target:

```bash
python -m pip install -e '.[llm]'
QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14 \
  python scripts/paper/run.py prepare_qwen3_layer14_artifacts
```

The paper defaults capture 500,000 paired rows from WikiText at layer 14 with
`d_model=1024`. The underlying command is:

```bash
python -m hebbian.expts.llm_embeddings.capture_mlp_io \
  --output-root /path/to/qwen3_06b_layer14 \
  --model-id Qwen/Qwen3-0.6B-Base \
  --layer-index 14 \
  --max-pairs 500000 \
  --seq-length 1024 \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --dataset-split train \
  --batch-size 4 \
  --device cuda \
  --save-dtype float32
```

Validate an existing bundle with:

```bash
QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14 \
QWEN3_CHECK_SCOPE=activations \
QWEN3_EXPECTED_D_MODEL=1024 \
QWEN3_MIN_ROWS=500000 \
QWEN3_RUN_MLP_CHECK=true \
  python scripts/paper/run.py check_qwen3_artifacts
```

For a CPU wiring check, use a small local text file:

```bash
QWEN3_TEXT_FILE=/path/to/sample.txt \
QWEN3_MAX_PAIRS=32 \
QWEN3_SEQ_LENGTH=16 \
QWEN3_ALLOW_SHORT=true \
QWEN3_DEVICE=cpu \
QWEN3_RUN_TRANSFORMER_CHECK=false \
  python scripts/paper/run.py prepare_qwen3_layer14_artifacts
```

## Run Qwen3 Experiments

With `activations/x.pt` and `activations/y.pt` present:

```bash
QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14 \
  python scripts/paper/run.py fig_llm_margin_qwen3_layer14 --mode run-and-plot

QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14 \
  python scripts/paper/run.py fig_llm_mlp_capacity_qwen3_layer14 --mode run-and-plot

QWEN3_ARTIFACT_ROOT=/path/to/qwen3_06b_layer14 \
  python scripts/paper/run.py fig_llm_transformer_capacity_qwen3_layer14 --mode run-and-plot
```

The MLP and transformer capacity targets use the paper's `0.98` success
threshold by default. Their runner settings can be overridden through the
environment variables in `scripts/paper/targets/qwen.py`.

## Copying Into A Paper Checkout

Use `--paper-root` to copy a generated target into the expected manuscript path:

```bash
python scripts/paper/run.py fig_rf_margin_m --mode plot-only \
  --paper-root /path/to/mlps-are-hebbians-paper
```

This is optional and does not change the generated artifact tree.
