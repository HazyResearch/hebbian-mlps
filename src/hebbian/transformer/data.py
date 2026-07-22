"""Vectorized batches for the paper's associative-recall task."""

from __future__ import annotations

import torch

from hebbian.data.synthetics.factsets import BijectiveMapping


MASK_INDEX = -100


def _mapping_to_tensor(
    mapping: BijectiveMapping, device: torch.device
) -> torch.Tensor:
    lookup = torch.zeros(
        max(mapping.inputs) + 1, dtype=torch.long, device=device
    )
    for key, value in zip(mapping.inputs, mapping.outputs):
        lookup[key] = value
    return lookup


class AssociativeRecallBatchGenerator:
    """Generate ``<junk> K <junk> Q V`` batches directly on a device."""

    def __init__(
        self,
        mapping: BijectiveMapping,
        num_facts: int,
        junk_vocab_size: int,
        min_seq_length: int,
        max_seq_length: int,
        batch_size: int,
        num_batches: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_facts = num_facts
        self.junk_vocab_size = junk_vocab_size
        self.junk_start = num_facts
        self.query_token = num_facts + junk_vocab_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.seq_length = 2 * max_seq_length + 3
        self._lookup = _mapping_to_tensor(mapping, device)

    def __len__(self) -> int:
        return self.num_batches

    def _generate_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self.batch_size
        inputs = torch.full(
            (batch_size, self.seq_length),
            MASK_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        targets = torch.full_like(inputs, MASK_INDEX)
        prefix_lengths = torch.randint(
            self.min_seq_length,
            self.max_seq_length + 1,
            (batch_size,),
            device=self.device,
        )
        suffix_lengths = torch.randint(
            self.min_seq_length,
            self.max_seq_length + 1,
            (batch_size,),
            device=self.device,
        )
        prefix_junk = torch.randint(
            self.junk_start,
            self.junk_start + self.junk_vocab_size,
            (batch_size, self.max_seq_length),
            dtype=torch.long,
            device=self.device,
        )
        suffix_junk = torch.randint(
            self.junk_start,
            self.junk_start + self.junk_vocab_size,
            (batch_size, self.max_seq_length),
            dtype=torch.long,
            device=self.device,
        )
        keys = torch.randint(
            0,
            self.num_facts,
            (batch_size,),
            dtype=torch.long,
            device=self.device,
        )
        values = self._lookup[keys]
        batch_indices = torch.arange(batch_size, device=self.device)
        positions = torch.arange(self.max_seq_length, device=self.device).unsqueeze(0)

        prefix_mask = positions < prefix_lengths.unsqueeze(1)
        inputs[:, : self.max_seq_length][prefix_mask] = prefix_junk[prefix_mask]
        inputs[batch_indices, prefix_lengths] = keys

        suffix_starts = prefix_lengths + 1
        for offset in range(self.max_seq_length):
            mask = offset < suffix_lengths
            if mask.any():
                inputs[batch_indices[mask], suffix_starts[mask] + offset] = (
                    suffix_junk[mask, offset]
                )

        query_positions = suffix_starts + suffix_lengths
        inputs[batch_indices, query_positions] = self.query_token
        targets[batch_indices, query_positions] = values
        inputs[batch_indices, query_positions + 1] = values
        return inputs, targets

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._generate_batch()


def create_associative_recall_batches(
    dataset_config,
    train_config,
    mapping: BijectiveMapping,
    device: torch.device,
) -> tuple[AssociativeRecallBatchGenerator, AssociativeRecallBatchGenerator]:
    """Build train and evaluation batch streams for associative recall."""

    kwargs = {
        "mapping": mapping,
        "num_facts": dataset_config.num_facts,
        "junk_vocab_size": dataset_config.junk_vocab_size,
        "min_seq_length": dataset_config.min_seq_length,
        "max_seq_length": dataset_config.max_seq_length,
        "batch_size": train_config.batch_size,
        "num_batches": train_config.steps_per_dataset,
        "device": device,
    }
    return (
        AssociativeRecallBatchGenerator(**kwargs),
        AssociativeRecallBatchGenerator(**kwargs),
    )


__all__ = [
    "AssociativeRecallBatchGenerator",
    "MASK_INDEX",
    "create_associative_recall_batches",
]
