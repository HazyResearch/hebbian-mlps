"""
Attention pre-train + GD MLP hidden-dim sweep.

Two-phase pipeline per experiment:
  1. Pre-train attention on associative recall (identity mapping).
  2. Construct GD MLP with a fresh random-permutation mapping (same embeddings).
  3. Insert the frozen MLP into the trained attention model.
  4. Evaluate the combined model on associative recall (random mapping).

The sweep axis is hidden_dim (log-spaced).  For each hidden_dim we run n_seeds seeds.

Reported metrics (compatible with plot_hidden_dim_sweep.py):
  - mlp_accuracy_mean/std          : GD MLP standalone accuracy
  - transformer_accuracy_mean/std  : combined (attention + MLP) accuracy
  - transformer_train_accuracy_*   : attention-only accuracy (repurposed axis)
  - gamma_min_mean/std             : MLP minimum margin

Usage:
    python -m hebbian.expts.hidden_dim.run
    python -m hebbian.expts.hidden_dim.run \\
        'n_seeds=1' 'junk_len=10' 'hidden_dim_max=2048'
"""

from __future__ import annotations

import copy
import datetime
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from hebbian.config import main as main_decorator, pydraclass

from hebbian.gpu_sweep import GridSearchConfig
from hebbian.gpu_sweep import GPUJobResult
from hebbian.gpu_sweep import run_grid_searches

from hebbian.transformer.config import (
    AssociativeRecallConfig,
    DatasetConfig,
    TrainingConfig,
    EmbeddingsConfig,
    TransformerConfig,
)
from hebbian.transformer.train_attention import train_attention
from hebbian.transformer.utils import insert_mlp_into_gpt, evaluate
from hebbian.transformer.data import create_associative_recall_batches
from hebbian.transformer.fact_store import build_fact_mlp
from hebbian.data.synthetics.factsets import (
    Factset,
    create_random_permutation_mapping,
)


# ---------------------------------------------------------------------------
# CLI-level experiment config
# ---------------------------------------------------------------------------

@pydraclass
class ExperimentConfig:
    """Configuration for attn-pretrain + Hebbian MLP hidden-dim sweep."""

    # Fixed model / data
    d_model: int = 128
    num_facts: int = 2048
    junk_len: int = 9
    junk_vocab_size: int = 9

    # MLP sweep
    mlp_method: str = "gd"
    hidden_dim_min: int = 16
    hidden_dim_max: int = 2048
    n_hidden_dims: int = 10      # log2-spaced points

    # Training
    num_epochs: int = 4000
    batch_size: int = 1280
    learning_rate: float = 2e-4
    steps_per_dataset: int = 1
    disable_early_stopping: bool = False
    attn_residual: bool = False
    freeze_voproj: bool = False

    # Seeds
    n_seeds: int = 4

    # Output
    base_dir: str | None = None
    success_threshold: float = 0.98

    # GPU scheduling
    max_gpus: int = 8
    simultaneous_jobs_per_gpu: int = 4


# ---------------------------------------------------------------------------
# Per-experiment config (picklable, passed through multiprocessing)
# ---------------------------------------------------------------------------

@dataclass
class AttnMlpConfig:
    """Config for a single attention-pretrain + MLP evaluation experiment."""

    attention_config: AssociativeRecallConfig
    mlp_method: str = "gd"
    mlp_hidden_dim: int = 256
    seed: int = 42
    save_dir: Optional[str] = None

    def finalize(self):
        """Called by the sweep runner before the experiment runs."""
        dc = self.attention_config.dataset_config
        tc = self.attention_config.train_config
        if hasattr(dc, "custom_finalize"):
            dc.custom_finalize()
        if hasattr(tc, "custom_finalize"):
            tc.custom_finalize()


# ---------------------------------------------------------------------------
# Helpers shared with train.py (duplicated to avoid circular imports)
# ---------------------------------------------------------------------------

def get_hidden_dims(low: int, high: int, n: int) -> List[int]:
    """Return n integers log2-spaced between low and high (inclusive)."""
    raw = np.logspace(np.log2(low), np.log2(high), n, base=2)
    return sorted(set(int(round(v)) for v in raw))


def _make_eval_factset(train_factset: Factset, seed: int) -> Factset:
    """Same embeddings, fresh random-permutation mapping."""
    eval_mapping = create_random_permutation_mapping(train_factset.vocab_size, seed=seed)
    eval_factset = object.__new__(Factset)
    eval_factset.input_embeddings = train_factset.input_embeddings
    eval_factset.output_embeddings = train_factset.output_embeddings
    eval_factset.mapping = eval_mapping
    eval_factset.d_model = train_factset.d_model
    eval_factset.vocab_size = train_factset.vocab_size
    return eval_factset


@torch.no_grad()
def _compute_per_key_margins(
    mlp: nn.Module,
    factset: Factset,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Per-key margin: γ_i = <mlp(k_i)/||·||, v_{f(i)}> - max_{j≠f(i)} <mlp(k_i)/||·||, v_j>.

    Returns array of shape (n_facts,).
    """
    mlp.eval()
    K = factset.input_embeddings.to(device=device, dtype=dtype)
    V_all = factset.output_embeddings.to(device=device, dtype=dtype)
    n = K.shape[0]

    value_idx = torch.tensor(
        [factset.mapping.get_output(i) for i in range(n)],
        device=device, dtype=torch.long,
    )

    Y = mlp(K)
    Y = Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scores = Y @ V_all.T

    batch = torch.arange(n, device=device)
    correct = scores[batch, value_idx]
    scores_masked = scores.clone()
    scores_masked[batch, value_idx] = float("-inf")
    max_wrong = scores_masked.max(dim=1).values

    return (correct - max_wrong).cpu().numpy()


@torch.no_grad()
def _compute_mlp_gamma_min(
    mlp: nn.Module,
    factset: Factset,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Minimum margin of normalised mlp(K) scored against value embeddings."""
    return float(_compute_per_key_margins(mlp, factset, device, dtype).min())


# ---------------------------------------------------------------------------
# Config factory for attention pre-training phase
# ---------------------------------------------------------------------------

def get_attention_config(
    d_model: int,
    num_facts: int,
    junk_len: int,
    junk_vocab_size: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    steps_per_dataset: int,
    disable_early_stopping: bool,
    attn_residual: bool,
    seed: int,
    freeze_voproj: bool = False,
) -> AssociativeRecallConfig:
    """Configuration for attention-only associative-recall pre-training."""
    return AssociativeRecallConfig(
        dataset_config=DatasetConfig(
            num_facts=num_facts,
            junk_vocab_size=junk_vocab_size,
            min_seq_length=junk_len,
            max_seq_length=junk_len,
            use_identity_fact_mapping=True,
        ),
        train_config=TrainingConfig(
            embeddings_config=EmbeddingsConfig(d_model=d_model),
            transformer_config=TransformerConfig(
                freeze_value_dense_identity=freeze_voproj,
                use_identity_mlp=True,
                no_positional_encoding=True,
                lm_head_norm_type="unit_rmsnorm",
                mlp_norm_type="unit_rmsnorm",
                attn_residual=attn_residual,
                freeze_input_embeddings=True,
                freeze_output_embeddings=True,
            ),
            batch_size=batch_size,
            lr=learning_rate,
            steps_per_dataset=steps_per_dataset,
            epochs=num_epochs,
            early_stop_accuracy=None if disable_early_stopping else 0.99,
            seed=seed,
            save_at_end=False,   # we save our own checkpoint below
            base_dir=None,       # suppress figs/checkpoints from phase 1
        ),
    )


# ---------------------------------------------------------------------------
# Two-phase experiment function
# ---------------------------------------------------------------------------

def train_attn_pretrain_mlp(config: AttnMlpConfig) -> Dict[str, Any]:
    """
    Phase 1: train attention on associative recall (identity mapping).
    Phase 2: construct Hebbian MLP on random-permutation mapping.
    Phase 3: insert frozen MLP into the trained attention model.
    Phase 4: evaluate the combined model on associative recall.
    """
    attn_cfg = config.attention_config
    dc = attn_cfg.dataset_config
    tc = attn_cfg.train_config
    seed = config.seed

    # Finalize configs
    if hasattr(dc, "custom_finalize"):
        dc.custom_finalize()
    if hasattr(tc, "custom_finalize"):
        tc.custom_finalize()

    device = torch.device(
        tc.device if torch.cuda.is_available() or tc.device == "cpu" else "cpu"
    )
    dtype = tc.dtype

    if config.save_dir is not None:
        os.makedirs(config.save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Phase 1: Train attention (associative recall, identity mapping)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Training attention on associative-recall task...")
    print("=" * 60)

    attn_results = train_attention(attn_cfg)
    gpt_model = attn_results["gpt_model"]
    factset = attn_results["factset"]
    attn_best_acc = attn_results["best_acc"]
    print(f"Attention best accuracy: {attn_best_acc:.4f}")

    # ----------------------------------------------------------------
    # Phase 2: Build the fact-storing MLP (random permutation, same embeddings)
    # via the configured method (default "gd"; also "hebbian", "ntk", ...).
    # ----------------------------------------------------------------
    print("=" * 60)
    print(f"Phase 2: Building {config.mlp_method} MLP (random permutation)...")
    print("=" * 60)

    mlp_factset = _make_eval_factset(factset, seed=seed + 7777)

    mlp_config = copy.deepcopy(attn_cfg)
    mlp_config.train_config.mlp_method = config.mlp_method
    mlp_config.train_config.mlp_hidden_dim = config.mlp_hidden_dim

    mlp, mlp_metrics = build_fact_mlp(mlp_config, mlp_factset)
    mlp = mlp.to(device=device, dtype=dtype)

    mlp_accuracy = mlp_metrics.get("accuracy", mlp_metrics.get("final_accuracy", float("nan")))
    print(f"  MLP accuracy: {mlp_accuracy}")

    per_key_margins = _compute_per_key_margins(mlp, mlp_factset, device, dtype)
    gamma_min = float(per_key_margins.min())
    mlp_metrics["gamma_min"] = gamma_min
    mlp_metrics["hidden_dim"] = config.mlp_hidden_dim
    mlp_metrics["per_key_margins"] = per_key_margins
    print(f"  MLP gamma_min: {gamma_min:.4f}")

    # ----------------------------------------------------------------
    # Phase 3: Insert frozen MLP into trained attention model
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Phase 3: Inserting MLP into attention model...")
    print("=" * 60)

    # Use wte already loaded in gpt_model (populated during phase 1)
    gpt_model.eval()
    insert_mlp_into_gpt(
        gpt_model,
        mlp,
        gpt_model.transformer.wte,
        freeze_mlp=True,
        freeze_wte=tc.transformer_config.freeze_input_embeddings,
        freeze_lm_head=tc.transformer_config.freeze_output_embeddings,
    )

    # ----------------------------------------------------------------
    # Phase 4: Evaluate associative recall with the random mapping
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Phase 4: Evaluating combined model...")
    print("=" * 60)

    _, eval_dataloader = create_associative_recall_batches(
        dc, tc, mlp_factset.mapping, device=device,
    )

    gpt_model.eval()
    num_iters = max(
        1,
        int(factset.vocab_size / (len(eval_dataloader) * eval_dataloader.batch_size)),
    ) * 10
    eval_result = evaluate(gpt_model, eval_dataloader, device, num_iterations=num_iters)
    combined_acc = eval_result["accuracy"]
    print(f"Combined model accuracy: {combined_acc:.4f}")

    # Capture architecture strings
    mlp_str = str(mlp)
    gpt_str = str(gpt_model)

    # ----------------------------------------------------------------
    # Save checkpoint
    # ----------------------------------------------------------------
    if config.save_dir is not None:
        checkpoint_path = os.path.join(config.save_dir, "last_model.pt")
        torch.save(
            {
                "best_acc": combined_acc,
                "best_train_acc": attn_best_acc,   # repurposed for plot
                "attn_best_acc": attn_best_acc,
                "mlp_metrics": mlp_metrics,
                "mlp_str": mlp_str,
                "gpt_str": gpt_str,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    return {
        "best_acc": combined_acc,
        "best_train_acc": attn_best_acc,   # repurposed: attention-only acc
        "attn_best_acc": attn_best_acc,
        "mlp_metrics": mlp_metrics,
        "mlp_str": mlp_str,
        "gpt_str": gpt_str,
    }


# ---------------------------------------------------------------------------
# Grid search config for one hidden_dim (seeds swept)
# ---------------------------------------------------------------------------

@pydraclass
class AttnPretrainGridSearchConfig(GridSearchConfig):
    """Grid search config for a single hidden_dim, sweeping seeds."""

    d_model: int = 128
    num_facts: int = 2048
    hidden_dim: int = 256
    mlp_method: str = "gd"

    def get_experiment_config_and_base_dir(self, **kwargs):
        seed = int(kwargs.get("seed", 42))

        # Deep-copy and set seed
        attn_cfg = copy.deepcopy(self.base_experiment_config)
        attn_cfg.train_config.seed = seed

        base_dir = f"{self.base_dir}/seed_{seed}"
        os.makedirs(base_dir, exist_ok=True)

        exp_config = AttnMlpConfig(
            attention_config=attn_cfg,
            mlp_method=self.mlp_method,
            mlp_hidden_dim=self.hidden_dim,
            seed=seed,
            save_dir=os.path.join(base_dir, "checkpoints"),
        )
        return exp_config, base_dir

    def run_experiment_config(self, config):
        return train_attn_pretrain_mlp(config)

    def agg_results(self, results: List[GPUJobResult]) -> Any:
        """Aggregate over seeds: collect metrics, return summary dict."""
        successful = [r for r in results if r.success and r.result is not None]
        if not successful:
            return {"error": "No successful results", "hidden_dim": self.hidden_dim}

        def _extract(r, *keys, default=float("nan")):
            obj = r.result
            for k in keys:
                if isinstance(obj, dict):
                    obj = obj.get(k, default)
                else:
                    return default
            return obj

        combined_accs = [_extract(r, "best_acc") for r in successful]
        attn_accs = [_extract(r, "attn_best_acc") for r in successful]

        mlp_accs = [
            _extract(r, "mlp_metrics", "accuracy")
            if not (
                isinstance(_extract(r, "mlp_metrics", "accuracy"), float)
                and np.isnan(_extract(r, "mlp_metrics", "accuracy"))
            )
            else _extract(r, "mlp_metrics", "final_accuracy")
            for r in successful
        ]
        gamma_mins = [_extract(r, "mlp_metrics", "gamma_min") for r in successful]
        param_count = _extract(successful[0], "mlp_metrics", "param_count")
        hd = _extract(successful[0], "mlp_metrics", "hidden_dim", default=self.hidden_dim)
        mlp_str = _extract(successful[0], "mlp_str", default=None)
        gpt_str = _extract(successful[0], "gpt_str", default=None)

        def _stat(vals):
            v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
            return (
                np.mean(v) if v else float("nan"),
                np.std(v) if v else float("nan"),
            )

        mlp_mean, mlp_std = _stat(mlp_accs)
        comb_mean, comb_std = _stat(combined_accs)
        attn_mean, attn_std = _stat(attn_accs)
        gmin_mean, gmin_std = _stat(gamma_mins)

        return {
            "hidden_dim": hd,
            "param_count": param_count,
            "mlp_accuracy_mean": mlp_mean,
            "mlp_accuracy_std": mlp_std,
            # combined model accuracy → "transformer" eval slot (for plot compat)
            "transformer_accuracy_mean": comb_mean,
            "transformer_accuracy_std": comb_std,
            # attention-only accuracy → "transformer train" slot (for plot compat)
            "transformer_train_accuracy_mean": attn_mean,
            "transformer_train_accuracy_std": attn_std,
            "gamma_min_mean": gmin_mean,
            "gamma_min_std": gmin_std,
            "n_seeds": len(successful),
            "mlp_str": mlp_str,
            "gpt_str": gpt_str,
            "all_results": [r.result for r in successful],
        }


# ---------------------------------------------------------------------------
# Build sweep configs
# ---------------------------------------------------------------------------

def get_sweep_configs(config: ExperimentConfig):
    if config.base_dir is None:
        ts = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
        base_dir_root = (
            "./artifacts/hidden_dim/run"
            f"_d{config.d_model}_n{config.num_facts}"
            f"_junk{config.junk_len}_{ts}"
        )
    else:
        base_dir_root = config.base_dir

    print(f"Base directory: {base_dir_root}")

    hidden_dims = get_hidden_dims(config.hidden_dim_min, config.hidden_dim_max, config.n_hidden_dims)
    seeds = [42 + i * 10000 for i in range(config.n_seeds)]

    print(f"Hidden dims ({len(hidden_dims)}): {hidden_dims}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    configs = []
    for m in hidden_dims:
        base_attn = get_attention_config(
            d_model=config.d_model,
            num_facts=config.num_facts,
            junk_len=config.junk_len,
            junk_vocab_size=config.junk_vocab_size,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            steps_per_dataset=config.steps_per_dataset,
            disable_early_stopping=config.disable_early_stopping,
            attn_residual=config.attn_residual,
            seed=42,  # overridden per-seed in get_experiment_config_and_base_dir
            freeze_voproj=config.freeze_voproj,
        )

        m_dir = os.path.join(base_dir_root, f"m{m}")
        os.makedirs(m_dir, exist_ok=True)

        grid_cfg = AttnPretrainGridSearchConfig(
            base_dir=m_dir,
            sweep_props={"seed": seeds},
            base_experiment_config=base_attn,
            d_model=config.d_model,
            num_facts=config.num_facts,
            hidden_dim=m,
            mlp_method=config.mlp_method,
        )
        configs.append(grid_cfg)

    return configs, base_dir_root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@main_decorator(ExperimentConfig)
def main(config: ExperimentConfig):
    configs, base_dir_root = get_sweep_configs(config)
    print(
        f"\nRunning {len(configs)} grid search configs "
        f"({sum(len(c.sweep_props['seed']) for c in configs)} total experiments):"
    )
    for cfg in configs:
        print(f"  m={cfg.hidden_dim}  dir={cfg.base_dir}")

    run_grid_searches(
        configs,
        max_gpus=config.max_gpus,
        simultaneous_jobs_per_gpu=config.simultaneous_jobs_per_gpu,
    )

    print(f"\nDone. Results saved to: {base_dir_root}")
    print("Plot with:")
    print(f"  python -m hebbian.expts.hidden_dim.plot_hidden_dim_sweep 'base_dir=\"{base_dir_root}\"'")


if __name__ == "__main__":
    main()
