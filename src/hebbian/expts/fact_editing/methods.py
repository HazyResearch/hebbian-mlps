"""Fact-editing methods for the repository's MLP layout."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from hebbian.data.synthetics.factsets import BijectiveMapping, Factset
from hebbian.expts.fact_editing.common import (
    construct_mlp,
    get_edit_mlp,
    get_mlp_edit_key_from_hook_input,
    get_mlp_edit_module,
    get_mlp_edit_weight,
    project_mlp_edit_residuals,
    set_edit_mlp,
    unwrap_edit_mlp,
)
from hebbian.transformer.utils import insert_mlp_into_gpt


_CUDA_LINEAR_ALG_ERROR_TOKENS = (
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "CUBLAS_STATUS_NOT_INITIALIZED",
    "cublasDtrsm",
    "cublasStrsm",
    "cublasSgemm",
    "CUSOLVER_STATUS_INTERNAL_ERROR",
    "cusolver error",
)


@dataclass
class EditMetrics:
    efficacy: float
    paraphrase: float
    specificity: float
    specificity_paraphrase: float
    non_fact_pre_nll: float = float("nan")
    non_fact_post_nll: float = float("nan")
    non_fact_pre_ppl: float = float("nan")
    non_fact_post_ppl: float = float("nan")
    non_fact_ppl_ratio: float = float("nan")
    non_fact_num_tokens: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _is_cuda_linalg_backend_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return any(token in msg for token in _CUDA_LINEAR_ALG_ERROR_TOKENS)


def _robust_linalg_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(A, B)
    except RuntimeError as exc:
        if A.device.type != "cuda" or not _is_cuda_linalg_backend_error(exc):
            raise
    solved_cpu = torch.linalg.solve(A.detach().cpu(), B.detach().cpu())
    return solved_cpu.to(device=A.device, dtype=A.dtype)


def _robust_linalg_svd(
    matrix: torch.Tensor,
    *,
    full_matrices: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.svd(matrix, full_matrices=full_matrices)
    except RuntimeError as exc:
        if matrix.device.type != "cuda" or not _is_cuda_linalg_backend_error(exc):
            raise
    u_cpu, s_cpu, vh_cpu = torch.linalg.svd(matrix.detach().cpu(), full_matrices=full_matrices)
    return (
        u_cpu.to(device=matrix.device, dtype=matrix.dtype),
        s_cpu.to(device=matrix.device, dtype=matrix.dtype),
        vh_cpu.to(device=matrix.device, dtype=matrix.dtype),
    )


def _make_permutation(num_facts: int, num_preserve_facts: int, num_alter_facts: int) -> np.ndarray:
    tail_start = num_preserve_facts + num_alter_facts
    return np.concatenate(
        [
            np.arange(num_preserve_facts),
            np.random.permutation(num_alter_facts) + num_preserve_facts,
            np.arange(tail_start, num_facts),
        ]
    )


def _permute_facts(
    facts: Sequence[Any],
    permutation: Sequence[int],
) -> List[Any]:
    return [
        type(facts[index])(book=facts[index].book, author=facts[permutation[index]].author)
        for index in range(len(facts))
    ]


def _permute_example_groups(
    example_groups: Sequence[Sequence[Dict[str, Any]]],
    permutation: Sequence[int],
) -> List[List[Dict[str, Any]]]:
    altered_groups: List[List[Dict[str, Any]]] = []
    for index, group in enumerate(example_groups):
        altered_group: List[Dict[str, Any]] = []
        target_group = example_groups[permutation[index]]
        target_value_token_ids = target_group[0]["value_token_ids"]
        for example in group:
            altered = copy.deepcopy(example)
            altered["labels_last"] = list(example["labels_last"])
            altered["labels_last"][:-len(target_value_token_ids)] = [-100] * max(
                0,
                len(altered["labels_last"]) - len(target_value_token_ids),
            )
            altered["labels_last"][-len(target_value_token_ids):] = list(target_value_token_ids)
            altered["value_token_ids"] = list(target_value_token_ids)
            if altered["labels"] == example["labels_last"]:
                altered["labels"] = list(altered["labels_last"])
            altered_group.append(altered)
        altered_groups.append(altered_group)
    return altered_groups


def get_datasets(
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
    facts: Sequence[Any],
    factset: Factset,
    num_preserve_facts: int,
    num_alter_facts: int,
    *,
    max_sentences: int = 1,
    seed: int = 42,
) -> tuple[Factset, tuple[list[list[dict]], list[list[dict]]], tuple[list[list[dict]], list[list[dict]]]]:
    set_seed(seed)
    permutation = _make_permutation(len(fact_groups), num_preserve_facts, num_alter_facts)
    altered_facts = _permute_facts(facts, permutation)
    altered_fact_groups = _permute_example_groups(fact_groups, permutation)

    selected_groups = list(altered_fact_groups[: num_preserve_facts + num_alter_facts])
    shuffled_groups: list[list[dict]] = []
    for group in selected_groups:
        order = np.random.permutation(len(group))
        shuffled_groups.append([copy.deepcopy(group[index]) for index in order])

    preserve_groups = shuffled_groups[:num_preserve_facts]
    alter_groups = shuffled_groups[num_preserve_facts:]

    train_preserve = [group[:max_sentences] for group in preserve_groups]
    train_alter = [group[:max_sentences] for group in alter_groups]
    test_preserve = [group[max_sentences:] for group in preserve_groups]
    test_alter = [group[max_sentences:] for group in alter_groups]

    altered_factset = Factset(
        input_embeddings=factset.input_embeddings,
        output_embeddings=factset.output_embeddings,
        mapping=BijectiveMapping(inputs=list(range(factset.vocab_size)), outputs=permutation.tolist()),
        d_model=factset.d_model,
        vocab_size=factset.vocab_size,
    )
    _ = altered_facts
    return altered_factset, (train_preserve, train_alter), (test_preserve, test_alter)


@torch.no_grad()
def collect_keys(gpt_model: nn.Module, fact_groups: Sequence[Sequence[Dict[str, Any]]], device: str) -> List[torch.Tensor]:
    feature_vectors: List[torch.Tensor] = []
    core_mlp = unwrap_edit_mlp(get_edit_mlp(gpt_model))
    edit_module = get_mlp_edit_module(core_mlp)

    for group in tqdm(fact_groups, desc="Collecting keys"):
        per_fact: List[torch.Tensor] = []
        for example in group:
            captured_inputs: List[torch.Tensor] = []

            def pre_hook(module, inputs):
                captured_inputs.append(inputs[0].detach())

            hook_handle = edit_module.register_forward_pre_hook(pre_hook)
            input_tensor = torch.tensor(example["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            labels_tensor = torch.tensor(example["labels_last"], dtype=torch.long, device=device).unsqueeze(0)
            _ = gpt_model(input_tensor, targets=labels_tensor)
            hook_handle.remove()
            if not captured_inputs:
                raise RuntimeError("Failed to collect activations for edit key extraction.")
            edit_key = get_mlp_edit_key_from_hook_input(core_mlp, captured_inputs[0])
            per_fact.append(edit_key[0, -1, :].cpu())
        feature_vectors.append(torch.cat(per_fact, dim=0))
    return feature_vectors


class ResidualAggregator(nn.Module):
    def __init__(self, layer: nn.Module, d_model: int):
        super().__init__()
        self.layer = layer
        self.residual = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        layer_output = self.layer(x)
        layer_output[:, -1, :] = layer_output[:, -1, :] + self.residual
        return layer_output


def optimize_residual(
    gpt_model: nn.Module,
    example: Dict[str, Any],
    device: str,
    d_model: int,
    *,
    num_steps: int = 100,
    lr: float = 0.01,
    wd: float = 0.0,
    early_stopping: float | None = None,
) -> torch.Tensor:
    original_mlp = get_edit_mlp(gpt_model)
    residual_aggregator = ResidualAggregator(original_mlp, d_model).to(device)
    set_edit_mlp(gpt_model, residual_aggregator)

    for param in gpt_model.parameters():
        param.requires_grad = False
    residual_aggregator.residual.requires_grad = True

    optimizer = torch.optim.Adam([residual_aggregator.residual], lr=lr, weight_decay=wd)
    input_tensor = torch.tensor(example["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    labels_tensor = torch.tensor(example["labels_last"], dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(num_steps):
        optimizer.zero_grad()
        _, loss = gpt_model(input_tensor, targets=labels_tensor)
        loss.backward()
        optimizer.step()
        if early_stopping is not None and loss.item() < early_stopping:
            break

    learned_residual = residual_aggregator.residual.detach().clone()
    set_edit_mlp(gpt_model, original_mlp)
    for param in gpt_model.parameters():
        param.requires_grad = True
    return learned_residual


def collect_desired_residuals(
    gpt_model: nn.Module,
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
    device: str,
    *,
    num_steps: int = 100,
    lr: float = 0.01,
    wd: float = 0.0,
    early_stopping: float | None = None,
) -> List[torch.Tensor]:
    d_model = gpt_model.config.n_embd
    residuals: List[torch.Tensor] = []
    for group in tqdm(fact_groups, desc="Optimizing residuals"):
        residuals.append(
            optimize_residual(
                gpt_model,
                group[0],
                device,
                d_model,
                num_steps=num_steps,
                lr=lr,
                wd=wd,
                early_stopping=early_stopping,
            )
        )
    return residuals


def clip_norm_residual(
    weight: torch.Tensor,
    alter_keys: torch.Tensor,
    residuals: torch.Tensor,
    clip_norm: float,
) -> torch.Tensor:
    original_values = weight @ alter_keys
    original_norms = torch.norm(original_values, dim=0)
    residual_norms = torch.norm(residuals, dim=0)
    clip_threshold = clip_norm * original_norms
    needs_clipping = residual_norms > clip_threshold
    clipped = residuals.clone()
    if needs_clipping.any():
        scaling = torch.where(needs_clipping, clip_threshold / residual_norms, 1.0)
        clipped = clipped * scaling.unsqueeze(0)
    return clipped


@torch.no_grad()
def evaluate_acc(gpt_model: nn.Module, fact_groups: Sequence[Sequence[Dict[str, Any]]]) -> float:
    total_correct = 0
    total_targets = 0
    device = next(gpt_model.parameters()).device
    for group in tqdm(fact_groups, desc="Evaluating"):
        for example in group:
            input_tensor = torch.tensor(example["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            labels_tensor = torch.tensor(example["labels_last"], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = gpt_model(input_tensor, targets=labels_tensor)
            predictions = logits.argmax(dim=-1)
            valid_mask = labels_tensor != -100
            total_correct += ((predictions == labels_tensor) & valid_mask).sum().item()
            total_targets += valid_mask.sum().item()
    return total_correct / total_targets if total_targets else 0.0


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def _get_non_fact_labels(example: Dict[str, Any]) -> List[int]:
    if "labels_all_but_last" in example:
        return list(example["labels_all_but_last"])
    labels = list(example.get("labels_full", example["labels"]))
    labels_last = list(example["labels_last"])
    for index, label in enumerate(labels_last):
        if label != -100:
            labels[index] = -100
    return labels


def _combined_eval_groups(
    train_preserve: Sequence[Sequence[Dict[str, Any]]],
    train_alter: Sequence[Sequence[Dict[str, Any]]],
    test_preserve: Sequence[Sequence[Dict[str, Any]]],
    test_alter: Sequence[Sequence[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    return [
        *[list(group) for group in train_preserve],
        *[list(group) for group in train_alter],
        *[list(group) for group in test_preserve],
        *[list(group) for group in test_alter],
    ]


def _empty_non_fact_ppl() -> Dict[str, float]:
    return {"nll": float("nan"), "ppl": float("nan"), "num_tokens": 0}


@torch.no_grad()
def evaluate_non_fact_ppl(
    gpt_model: nn.Module,
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
) -> Dict[str, float]:
    total_loss = 0.0
    total_targets = 0
    device = next(gpt_model.parameters()).device
    for group in tqdm(fact_groups, desc="Evaluating non-fact PPL"):
        for example in group:
            input_tensor = torch.tensor(example["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            labels_tensor = torch.tensor(
                _get_non_fact_labels(example),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            valid_targets = int((labels_tensor != -100).sum().item())
            if valid_targets == 0:
                continue
            logits, _ = gpt_model(input_tensor, targets=labels_tensor)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels_tensor.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss_sum.item())
            total_targets += valid_targets
    if total_targets == 0:
        return {"nll": float("nan"), "ppl": float("nan"), "num_tokens": 0}
    nll = total_loss / total_targets
    return {"nll": nll, "ppl": _safe_exp(nll), "num_tokens": total_targets}


def _build_edit_metrics(
    gpt_model: nn.Module,
    train_preserve: Sequence[Sequence[Dict[str, Any]]],
    train_alter: Sequence[Sequence[Dict[str, Any]]],
    test_preserve: Sequence[Sequence[Dict[str, Any]]],
    test_alter: Sequence[Sequence[Dict[str, Any]]],
    non_fact_pre: Dict[str, float],
    *,
    compute_non_fact_ppl: bool = True,
) -> EditMetrics:
    non_fact_post = (
        evaluate_non_fact_ppl(
            gpt_model,
            _combined_eval_groups(train_preserve, train_alter, test_preserve, test_alter),
        )
        if compute_non_fact_ppl
        else _empty_non_fact_ppl()
    )
    pre_ppl = non_fact_pre["ppl"]
    post_ppl = non_fact_post["ppl"]
    ppl_ratio = (
        post_ppl / pre_ppl
        if math.isfinite(pre_ppl) and pre_ppl > 0
        else float("nan")
    )
    return EditMetrics(
        efficacy=evaluate_acc(gpt_model, train_alter),
        paraphrase=evaluate_acc(gpt_model, test_alter),
        specificity=evaluate_acc(gpt_model, train_preserve),
        specificity_paraphrase=evaluate_acc(gpt_model, test_preserve),
        non_fact_pre_nll=non_fact_pre["nll"],
        non_fact_post_nll=non_fact_post["nll"],
        non_fact_pre_ppl=pre_ppl,
        non_fact_post_ppl=post_ppl,
        non_fact_ppl_ratio=ppl_ratio,
        non_fact_num_tokens=int(non_fact_post["num_tokens"]),
    )


def _collect_common_edit_inputs(
    gpt_model: nn.Module,
    facts: Sequence[Any],
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
    factset: Factset,
    num_preserve_facts: int,
    num_alter_facts: int,
    *,
    seed: int,
):
    altered_factset, (train_preserve, train_alter), (test_preserve, test_alter) = get_datasets(
        fact_groups,
        facts,
        factset,
        num_preserve_facts,
        num_alter_facts,
        max_sentences=1,
        seed=seed,
    )
    return altered_factset, train_preserve, train_alter, test_preserve, test_alter


@torch.no_grad()
def _compute_memit_weight(
    gpt_model: nn.Module,
    preserve_keys: List[torch.Tensor],
    alter_keys: List[torch.Tensor],
    alter_residuals: List[torch.Tensor],
    *,
    lambd: float,
    clip_norm: float | None,
    device: str,
) -> torch.Tensor:
    edit_mlp = get_edit_mlp(gpt_model)
    weight = get_mlp_edit_weight(edit_mlp).detach().clone().to(device)
    dtype = weight.dtype
    residuals = torch.stack(
        project_mlp_edit_residuals(edit_mlp, alter_residuals, device=device, dtype=dtype),
        dim=0,
    ).T
    alter_matrix = torch.stack(alter_keys, dim=0).T.to(device=device, dtype=dtype)
    preserve_matrix = torch.stack(preserve_keys, dim=0).T.to(device=device, dtype=dtype)
    preserve_cov = lambd * (preserve_matrix @ preserve_matrix.T) / preserve_matrix.shape[1]
    if clip_norm is not None:
        residuals = clip_norm_residual(weight, alter_matrix, residuals, clip_norm)
    solve_matrix = preserve_cov + alter_matrix @ alter_matrix.T
    delta = _robust_linalg_solve(solve_matrix.T, (residuals @ alter_matrix.T).T).T
    return weight + delta


def edit_factset_memit(
    gpt_model: nn.Module,
    facts: Sequence[Any],
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
    factset: Factset,
    *,
    num_preserve_facts: int,
    num_alter_facts: int,
    num_steps: int,
    lr: float,
    lambd: float,
    device: str,
    clip_norm: float | None,
    early_stopping: float | None,
    seed: int,
    compute_non_fact_ppl: bool = True,
) -> EditMetrics:
    _, train_preserve, train_alter, test_preserve, test_alter = _collect_common_edit_inputs(
        gpt_model,
        facts,
        fact_groups,
        factset,
        num_preserve_facts,
        num_alter_facts,
        seed=seed,
    )
    non_fact_pre = (
        evaluate_non_fact_ppl(
            gpt_model,
            _combined_eval_groups(train_preserve, train_alter, test_preserve, test_alter),
        )
        if compute_non_fact_ppl
        else _empty_non_fact_ppl()
    )
    preserve_keys = collect_keys(gpt_model, train_preserve, device)
    alter_keys = collect_keys(gpt_model, train_alter, device)
    alter_residuals = collect_desired_residuals(
        gpt_model,
        train_alter,
        device,
        num_steps=num_steps,
        lr=lr,
        early_stopping=early_stopping,
    )
    new_weight = _compute_memit_weight(
        gpt_model,
        preserve_keys,
        alter_keys,
        alter_residuals,
        lambd=lambd,
        clip_norm=clip_norm,
        device=device,
    )
    get_mlp_edit_weight(get_edit_mlp(gpt_model)).data.copy_(new_weight)
    return _build_edit_metrics(
        gpt_model,
        train_preserve,
        train_alter,
        test_preserve,
        test_alter,
        non_fact_pre,
        compute_non_fact_ppl=compute_non_fact_ppl,
    )


def get_nullspace_projector(matrix: torch.Tensor, tol: float) -> torch.Tensor:
    gram = matrix @ matrix.T
    singular_vectors, singular_values, _ = _robust_linalg_svd(gram, full_matrices=False)
    nullspace = singular_vectors[:, singular_values < tol]
    return nullspace @ nullspace.T


@torch.no_grad()
def _compute_alpha_weight(
    gpt_model: nn.Module,
    preserve_keys: List[torch.Tensor],
    alter_keys: List[torch.Tensor],
    alter_residuals: List[torch.Tensor],
    *,
    clip_norm: float | None,
    tol: float,
    device: str,
) -> torch.Tensor:
    edit_mlp = get_edit_mlp(gpt_model)
    weight = get_mlp_edit_weight(edit_mlp).detach().clone().to(device)
    dtype = weight.dtype
    residuals = torch.stack(
        project_mlp_edit_residuals(edit_mlp, alter_residuals, device=device, dtype=dtype),
        dim=0,
    ).T
    alter_matrix = torch.stack(alter_keys, dim=0).T.to(device=device, dtype=dtype)
    preserve_matrix = torch.stack(preserve_keys, dim=0).T.to(device=device, dtype=dtype)
    projector = get_nullspace_projector(preserve_matrix, tol=tol)
    if clip_norm is not None:
        residuals = clip_norm_residual(weight, alter_matrix, residuals, clip_norm)
    identity = torch.eye(alter_matrix.shape[0], dtype=alter_matrix.dtype, device=alter_matrix.device)
    solve_matrix = alter_matrix @ alter_matrix.T @ projector + identity
    rhs = (residuals @ alter_matrix.T @ projector).T
    delta = _robust_linalg_solve(solve_matrix.T, rhs).T
    return weight + delta


def edit_factset_alpha_edit(
    gpt_model: nn.Module,
    facts: Sequence[Any],
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
    factset: Factset,
    *,
    num_preserve_facts: int,
    num_alter_facts: int,
    num_steps: int,
    lr: float,
    device: str,
    clip_norm: float | None,
    early_stopping: float | None,
    tol: float,
    seed: int,
    compute_non_fact_ppl: bool = True,
) -> EditMetrics:
    _, train_preserve, train_alter, test_preserve, test_alter = _collect_common_edit_inputs(
        gpt_model,
        facts,
        fact_groups,
        factset,
        num_preserve_facts,
        num_alter_facts,
        seed=seed,
    )
    non_fact_pre = (
        evaluate_non_fact_ppl(
            gpt_model,
            _combined_eval_groups(train_preserve, train_alter, test_preserve, test_alter),
        )
        if compute_non_fact_ppl
        else _empty_non_fact_ppl()
    )
    preserve_keys = collect_keys(gpt_model, train_preserve, device)
    alter_keys = collect_keys(gpt_model, train_alter, device)
    alter_residuals = collect_desired_residuals(
        gpt_model,
        train_alter,
        device,
        num_steps=num_steps,
        lr=lr,
        early_stopping=early_stopping,
    )
    new_weight = _compute_alpha_weight(
        gpt_model,
        preserve_keys,
        alter_keys,
        alter_residuals,
        clip_norm=clip_norm,
        tol=tol,
        device=device,
    )
    get_mlp_edit_weight(get_edit_mlp(gpt_model)).data.copy_(new_weight)
    return _build_edit_metrics(
        gpt_model,
        train_preserve,
        train_alter,
        test_preserve,
        test_alter,
        non_fact_pre,
        compute_non_fact_ppl=compute_non_fact_ppl,
    )


@torch.no_grad()
def _compute_rome_weight(
    gpt_model: nn.Module,
    preserve_keys: List[torch.Tensor],
    alter_keys: List[torch.Tensor],
    alter_residuals: List[torch.Tensor],
    *,
    device: str,
) -> torch.Tensor:
    edit_mlp = get_edit_mlp(gpt_model)
    weight = get_mlp_edit_weight(edit_mlp).detach().clone().to(device)
    dtype = weight.dtype
    residuals = torch.stack(
        project_mlp_edit_residuals(edit_mlp, alter_residuals, device=device, dtype=dtype),
        dim=0,
    ).T
    alter_matrix = torch.stack(alter_keys, dim=0).T.to(device=device, dtype=dtype)
    preserve_matrix = torch.stack(preserve_keys, dim=0).T.to(device=device, dtype=dtype)
    cov = (preserve_matrix @ preserve_matrix.T) / preserve_matrix.shape[1]
    for residual, key in zip(residuals.T, alter_matrix.T):
        if residual.ndim == 1:
            residual = residual.unsqueeze(1)
        if key.ndim == 1:
            key = key.unsqueeze(1)
        solved = _robust_linalg_solve(cov, key)
        denom = (key.T @ solved).squeeze()
        scale = residual / denom
        weight = weight + scale @ solved.T
    return weight


def edit_factset_rome(
    gpt_model: nn.Module,
    facts: Sequence[Any],
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
    factset: Factset,
    *,
    num_preserve_facts: int,
    num_alter_facts: int,
    num_steps: int,
    lr: float,
    device: str,
    early_stopping: float | None,
    wd: float,
    seed: int,
    compute_non_fact_ppl: bool = True,
) -> EditMetrics:
    _, train_preserve, train_alter, test_preserve, test_alter = _collect_common_edit_inputs(
        gpt_model,
        facts,
        fact_groups,
        factset,
        num_preserve_facts,
        num_alter_facts,
        seed=seed,
    )
    non_fact_pre = (
        evaluate_non_fact_ppl(
            gpt_model,
            _combined_eval_groups(train_preserve, train_alter, test_preserve, test_alter),
        )
        if compute_non_fact_ppl
        else _empty_non_fact_ppl()
    )
    preserve_keys = collect_keys(gpt_model, train_preserve, device)
    alter_keys = collect_keys(gpt_model, train_alter, device)
    alter_residuals = collect_desired_residuals(
        gpt_model,
        train_alter,
        device,
        num_steps=num_steps,
        lr=lr,
        wd=wd,
        early_stopping=early_stopping,
    )
    new_weight = _compute_rome_weight(
        gpt_model,
        preserve_keys,
        alter_keys,
        alter_residuals,
        device=device,
    )
    get_mlp_edit_weight(get_edit_mlp(gpt_model)).data.copy_(new_weight)
    return _build_edit_metrics(
        gpt_model,
        train_preserve,
        train_alter,
        test_preserve,
        test_alter,
        non_fact_pre,
        compute_non_fact_ppl=compute_non_fact_ppl,
    )


def edit_factset_gd_construction(
    gpt_model: nn.Module,
    facts: Sequence[Any],
    fact_groups: Sequence[Sequence[Dict[str, Any]]],
    factset: Factset,
    full_embeddings: nn.Embedding,
    *,
    num_preserve_facts: int,
    num_alter_facts: int,
    method_name: str,
    hidden_dim: int | None,
    seed: int,
    device: str,
    method_kwargs: Dict[str, Any] | None,
    factset_dtype: torch.dtype | None = None,
    compute_non_fact_ppl: bool = True,
) -> EditMetrics:
    altered_factset, train_preserve, train_alter, test_preserve, test_alter = _collect_common_edit_inputs(
        gpt_model,
        facts,
        fact_groups,
        factset,
        num_preserve_facts,
        num_alter_facts,
        seed=seed,
    )
    non_fact_pre = (
        evaluate_non_fact_ppl(
            gpt_model,
            _combined_eval_groups(train_preserve, train_alter, test_preserve, test_alter),
        )
        if compute_non_fact_ppl
        else _empty_non_fact_ppl()
    )
    if factset_dtype is not None:
        altered_factset = altered_factset.to(device=device, dtype=factset_dtype)
    altered_mlp, _ = construct_mlp(
        factset=altered_factset,
        method_name=method_name,
        hidden_dim=hidden_dim,
        seed=seed,
        device=device,
        method_kwargs=method_kwargs,
    )
    altered_mlp = altered_mlp.to(device=device, dtype=next(gpt_model.parameters()).dtype)
    gpt_model.eval()
    insert_mlp_into_gpt(
        gpt_model,
        altered_mlp,
        full_embeddings,
        freeze_mlp=True,
        freeze_wte=True,
        freeze_lm_head=True,
    )
    return _build_edit_metrics(
        gpt_model,
        train_preserve,
        train_alter,
        test_preserve,
        test_alter,
        non_fact_pre,
        compute_non_fact_ppl=compute_non_fact_ppl,
    )
