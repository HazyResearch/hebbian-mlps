"""Train a base model for the author fact-editing experiments."""

from __future__ import annotations

import os
import copy
from typing import Any, Dict, List, Sequence

import torch
import torch.distributed as dist
from hebbian.config import main as main_decorator
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from hebbian.data.embeddings import generate_embeddings
from hebbian.data.language.authors import (
    AuthorExampleDataset,
    augment_tokenizer_with_author_facts,
    author_collate_fn,
    build_author_example_groups,
    build_author_factset,
    flatten_author_example_groups,
    load_author_facts,
)
from hebbian.expts.fact_editing.common import (
    construct_mlp,
    make_embedding_module,
    save_base_artifacts,
    set_random_seed,
    to_serializable,
    torch_dtype_to_name,
)
from hebbian.expts.fact_editing.config import BaseTrainConfig
from hebbian.transformer.model import GPT
from hebbian.transformer.utils import copy_embeddings_to_gpt, create_gpt_config, insert_mlp_into_gpt


class _EvalExampleDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        return {
            "input_ids": example["input_ids"],
            "labels_full": example["labels_full"],
            "labels_last": example["labels_last"],
        }


def _eval_collate_fn(features: Sequence[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_tensors = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
    label_full_tensors = [torch.tensor(feature["labels_full"], dtype=torch.long) for feature in features]
    label_last_tensors = [torch.tensor(feature["labels_last"], dtype=torch.long) for feature in features]
    batch_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=-100)
    batch_labels_full = pad_sequence(label_full_tensors, batch_first=True, padding_value=-100)
    batch_labels_last = pad_sequence(label_last_tensors, batch_first=True, padding_value=-100)
    return batch_inputs, batch_labels_full, batch_labels_last


def _unwrap_model(gpt_model: GPT | DistributedDataParallel) -> GPT:
    if isinstance(gpt_model, DistributedDataParallel):
        return gpt_model.module
    return gpt_model


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_main_process() -> bool:
    return _get_rank() == 0


def _setup_device(requested_device: str) -> torch.device:
    if _is_distributed():
        backend = "nccl" if requested_device != "cpu" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        if requested_device == "cpu":
            return torch.device("cpu")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available() or requested_device == "cpu":
        return torch.device(requested_device)
    return torch.device("cpu")


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _construct_mlp_once(
    *,
    factset: Any,
    method_name: str,
    hidden_dim: int | None,
    seed: int,
    device: str,
    method_kwargs: Dict[str, Any] | None,
) -> tuple[nn.Module, Dict[str, Any]]:
    if not _is_distributed():
        return construct_mlp(
            factset=factset,
            method_name=method_name,
            hidden_dim=hidden_dim,
            seed=seed,
            device=device,
            method_kwargs=method_kwargs,
        )

    payload: list[Any] = [None, None]
    if _is_main_process():
        mlp, mlp_metrics = construct_mlp(
            factset=factset,
            method_name=method_name,
            hidden_dim=hidden_dim,
            seed=seed,
            device=device,
            method_kwargs=method_kwargs,
        )
        payload[0] = mlp.cpu()
        payload[1] = mlp_metrics
    dist.broadcast_object_list(payload, src=0)
    return payload[0], payload[1]


def _evaluate_loader(
    gpt_model: GPT | DistributedDataParallel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    sync_across_ranks: bool,
) -> Dict[str, float]:
    gpt_model.eval()
    stats = torch.zeros(4, device=device, dtype=torch.float64)
    with torch.inference_mode():
        for inputs, labels_full, labels_last in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels_full = labels_full.to(device, non_blocking=True)
            labels_last = labels_last.to(device, non_blocking=True)
            logits, loss = gpt_model(inputs, targets=labels_full)
            batch_size = inputs.size(0)
            stats[0] += loss.detach().to(dtype=torch.float64) * batch_size
            stats[1] += batch_size
            preds = logits.argmax(dim=-1)
            valid_mask = labels_last != -100
            stats[2] += ((preds == labels_last) & valid_mask).sum().to(dtype=torch.float64)
            stats[3] += valid_mask.sum().to(dtype=torch.float64)
    if sync_across_ranks and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return {
        "loss": (stats[0] / max(stats[1].item(), 1.0)).item(),
        "value_accuracy": (stats[2] / max(stats[3].item(), 1.0)).item(),
    }


def run(config: BaseTrainConfig) -> Dict[str, Any]:
    config.finalize()
    tc = config.train_config
    set_random_seed(tc.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = _setup_device(tc.device)
    is_distributed = _is_distributed()
    is_main_process = _is_main_process()
    pin_memory = device.type == "cuda"
    os.makedirs(config.experiment_dir, exist_ok=True)
    os.makedirs(tc.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    facts = load_author_facts(config.authors_csv_path, num_facts=config.num_facts)
    book_component_token_ids = None
    if config.fact_input_embedding_mode == "average_compound_normalized":
        book_component_token_ids = [
            tokenizer.encode(f" {fact.book}", add_special_tokens=False)
            for fact in facts
        ]
    added_tokens = augment_tokenizer_with_author_facts(tokenizer, facts)

    train_dtype = tc.dtype
    mlp_dtype = tc.mlp_dtype or train_dtype
    d_model = tc.embeddings_config.d_model

    full_input_weights = generate_embeddings(
        len(tokenizer),
        d_model,
        init_type=tc.embeddings_config.embedding_init,
        dtype=train_dtype,
        device="cpu",
        seed=tc.seed,
    )
    if tc.embeddings_config.tie_embeddings:
        full_output_weights = full_input_weights.clone()
    else:
        full_output_weights = generate_embeddings(
            len(tokenizer),
            d_model,
            init_type=tc.embeddings_config.embedding_init,
            dtype=train_dtype,
            device="cpu",
            seed=tc.seed + 1,
        )

    factset, fact_token_info = build_author_factset(
        tokenizer,
        facts,
        full_input_weights,
        full_output_weights,
        factset_dtype=mlp_dtype,
        fact_input_embedding_mode=config.fact_input_embedding_mode,
        book_component_token_ids=book_component_token_ids,
    )
    if (
        config.fact_input_embedding_mode == "average_compound_normalized"
        and config.overwrite_compound_token_embeddings
    ):
        book_token_ids = fact_token_info["book_token_ids"]
        full_input_weights[book_token_ids] = factset.input_embeddings.to(
            dtype=full_input_weights.dtype,
            device=full_input_weights.device,
        )
        if tc.embeddings_config.tie_embeddings:
            full_output_weights = full_input_weights.clone()

    mlp, mlp_metrics = _construct_mlp_once(
        factset=factset,
        method_name=tc.mlp_method,
        hidden_dim=tc.mlp_hidden_dim,
        seed=tc.seed,
        device=str(device),
        method_kwargs=tc.mlp_method_kwargs,
    )
    mlp = mlp.to(device=device, dtype=train_dtype)

    example_groups = build_author_example_groups(
        tokenizer,
        facts,
        num_rephrases=config.num_rephrases,
        train_last_token=config.train_last_token,
    )
    flat_examples = flatten_author_example_groups(example_groups)
    dataset = AuthorExampleDataset(flat_examples, label_key="labels")
    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=tc.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=author_collate_fn,
        num_workers=config.train_num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.train_num_workers > 0,
    )
    eval_dataset = _EvalExampleDataset(flat_examples)
    eval_sampler = None
    if is_distributed:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=_eval_collate_fn,
        num_workers=config.eval_num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.eval_num_workers > 0,
    )

    max_seq_len = max(len(example["input_ids"]) for example in flat_examples)
    gpt_config = create_gpt_config(
        tc,
        type(
            "DatasetShim",
            (),
            {
                "block_size": max_seq_len + 2,
                "vocab_size": len(tokenizer),
            },
        )(),
    )
    gpt_model = GPT(gpt_config)
    gpt_model.to(device=device, dtype=train_dtype)

    full_input_embeddings = make_embedding_module(full_input_weights).to(device=device, dtype=train_dtype)
    full_output_embeddings = make_embedding_module(full_output_weights).to(device=device, dtype=train_dtype)
    fact_input_embeddings_for_norm = make_embedding_module(
        factset.input_embeddings.detach().to(dtype=train_dtype)
    ).to(device=device, dtype=train_dtype)
    copy_embeddings_to_gpt(
        gpt_model,
        full_input_embeddings,
        output_embeddings=full_output_embeddings,
        freeze_wte=tc.transformer_config.freeze_input_embeddings,
        freeze_lm_head=tc.transformer_config.freeze_output_embeddings,
        tie_embeddings=tc.embeddings_config.tie_embeddings,
    )

    gpt_model.eval()
    insert_mlp_into_gpt(
        gpt_model,
        mlp,
        fact_input_embeddings_for_norm,
        freeze_mlp=True,
        freeze_wte=tc.transformer_config.freeze_input_embeddings,
        freeze_lm_head=tc.transformer_config.freeze_output_embeddings,
    )

    if is_distributed:
        ddp_kwargs: Dict[str, Any] = {"find_unused_parameters": False}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        gpt_model = DistributedDataParallel(gpt_model, **ddp_kwargs)

    optimizer = _unwrap_model(gpt_model).configure_optimizers(
        weight_decay=tc.weight_decay,
        learning_rate=tc.lr,
        betas=(0.9, 0.999),
        device_type=device.type,
    )

    history: List[Dict[str, Any]] = []
    best_value_accuracy = float("-inf")
    best_state_dict = None

    for epoch in tqdm(range(tc.epochs), desc="Training base model", disable=not is_main_process):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        gpt_model.train()
        epoch_loss = 0.0
        batches = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            _, loss = gpt_model(inputs, targets=labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        should_eval = (epoch % tc.evaluate_every == 0) or (epoch == tc.epochs - 1)
        if should_eval:
            metrics = _evaluate_loader(
                gpt_model,
                eval_dataloader,
                device,
                sync_across_ranks=is_distributed,
            )
            epoch_loss_stats = torch.tensor([epoch_loss, batches], device=device, dtype=torch.float64)
            if is_distributed:
                dist.all_reduce(epoch_loss_stats, op=dist.ReduceOp.SUM)
            metrics["epoch"] = epoch
            metrics["train_loss"] = (epoch_loss_stats[0] / max(epoch_loss_stats[1].item(), 1.0)).item()
            if is_main_process:
                history.append(metrics)
            if is_main_process and metrics["value_accuracy"] > best_value_accuracy:
                best_value_accuracy = metrics["value_accuracy"]
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in _unwrap_model(gpt_model).state_dict().items()
                }
            if tc.early_stop_accuracy is not None and metrics["value_accuracy"] >= tc.early_stop_accuracy:
                break

    if is_distributed:
        dist.barrier()

    result = {
        "experiment_dir": config.experiment_dir,
        "final_metrics": None,
        "best_value_accuracy": best_value_accuracy,
    }
    if is_main_process:
        model_to_save = _unwrap_model(gpt_model)
        if best_state_dict is not None:
            model_to_save.load_state_dict(best_state_dict)
        full_eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=_eval_collate_fn,
            num_workers=config.eval_num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.eval_num_workers > 0,
        )
        final_metrics = _evaluate_loader(
            model_to_save,
            full_eval_dataloader,
            device,
            sync_across_ranks=False,
        )
        checkpoint_payload = {
            "epoch": history[-1]["epoch"] if history else 0,
            "gpt_state_dict": {
                key: value.detach().cpu()
                for key, value in model_to_save.state_dict().items()
            },
            "mlp_module": copy.deepcopy(mlp).cpu(),
            "gpt_config": to_serializable(gpt_config),
            "train_config": to_serializable(tc),
            "base_train_config": to_serializable(config),
            "mlp_metrics": to_serializable(mlp_metrics),
            "history": to_serializable(history),
            "final_metrics": to_serializable(final_metrics),
            "best_value_accuracy": best_value_accuracy,
        }
        embeddings_payload = {
            "full_input_embeddings": full_input_weights.detach().cpu(),
            "full_output_embeddings": full_output_weights.detach().cpu(),
            "fact_input_embeddings": factset.input_embeddings.detach().cpu(),
            "fact_output_embeddings": factset.output_embeddings.detach().cpu(),
            "mapping_outputs": list(factset.mapping.outputs),
            **fact_token_info,
        }
        metadata_payload = {
            "authors_csv_path": config.authors_csv_path,
            "tokenizer_name": config.tokenizer_name,
            "mlp_variant_label": config.mlp_variant_label or tc.mlp_method,
            "added_tokens": added_tokens,
            "facts": to_serializable(facts),
            "num_rephrases": config.num_rephrases,
            "train_last_token": config.train_last_token,
            "fact_input_embedding_mode": config.fact_input_embedding_mode,
            "overwrite_compound_token_embeddings": config.overwrite_compound_token_embeddings,
            "train_num_workers": config.train_num_workers,
            "eval_batch_size": config.eval_batch_size,
            "eval_num_workers": config.eval_num_workers,
            "train_dtype": torch_dtype_to_name(train_dtype),
            "mlp_dtype": torch_dtype_to_name(mlp_dtype),
            "mlp_method": tc.mlp_method,
            "mlp_hidden_dim": tc.mlp_hidden_dim,
            "mlp_method_kwargs": to_serializable(tc.mlp_method_kwargs),
            "seed": tc.seed,
            "n_layers": tc.transformer_config.n_layers,
            "use_mlp_qk": tc.transformer_config.use_mlp_qk,
            "transformer_bias": tc.transformer_config.bias,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }
        save_base_artifacts(
            experiment_dir=config.experiment_dir,
            checkpoint_payload=checkpoint_payload,
            embeddings_payload=embeddings_payload,
            metadata_payload=metadata_payload,
        )
        result["final_metrics"] = final_metrics

    _cleanup_distributed()
    return result


@main_decorator(BaseTrainConfig)
def main(config: BaseTrainConfig):
    run(config)


if __name__ == "__main__":
    main()
