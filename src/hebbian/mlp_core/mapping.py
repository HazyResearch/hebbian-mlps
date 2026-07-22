"""Mapping utilities for MLP construction.

This module provides classes for representing input-output mappings
used in MLP construction tasks.
"""

import math
import random
import numpy as np


def get_outputs_to_input(outputs: list[int]) -> list[int]:
    outputs_to_input = [0] * len(outputs)
    for i, output in enumerate(outputs):
        outputs_to_input[output] = i
    return outputs_to_input


def get_single_prefix_mappings(mapping: 'Mapping', prefix_suffix_correspondence: 'PrefixSuffixCorrespondence') -> list['SinglePrefixMapping']:
    single_prefix_mappings = [[None for _ in range(prefix_suffix_correspondence.max_suffix_indices[i] + 1)] for i in range(prefix_suffix_correspondence.max_prefix_index + 1)]
    for i, (prefix, suffix) in enumerate(prefix_suffix_correspondence):
        single_prefix_mappings[prefix][suffix] = mapping.get_output(i)

    for i in range(len(single_prefix_mappings)):
        for j in range(len(single_prefix_mappings[i])):
            assert single_prefix_mappings[i][j] is not None

    return [SinglePrefixMapping(prefix_mapping) for prefix_mapping in single_prefix_mappings]


def get_single_prefix_mappings_from_bins(mapping: 'Mapping', bins: list[list[int]]) -> list['SinglePrefixMapping']:
    prefix_mappings = [
        SinglePrefixMapping([
            mapping.get_output(item.item())
            for item in single_bin
        ])
        for single_bin in bins
    ]
    return prefix_mappings


class Mapping:
    def __init__(self, outputs: list[int], debug: bool = False):
        self.outputs = outputs
        self.outputs_to_input = get_outputs_to_input(outputs)

    @classmethod
    def generate_random_mapping(cls, num_mappings: int, debug: bool = False, seed: int = None) -> "Mapping":
        if seed is not None:
            rng = random.Random(seed)
            outputs = [rng.randint(0, num_mappings - 1) for _ in range(num_mappings)]
        else:
            outputs = [random.randint(0, num_mappings - 1) for _ in range(num_mappings)]

        return Mapping(outputs, debug=debug)

    @classmethod
    def generate_identity_mapping(
        cls, num_mappings: int, debug: bool = False
    ) -> "Mapping":
        outputs = [i for i in range(num_mappings)]

        return Mapping(outputs, debug=debug)

    @classmethod
    def generate_random_permutation_mapping(
        cls, num_mappings: int, debug: bool = False, seed: int = None
    ) -> "Mapping":
        if seed is not None:
            rng = np.random.default_rng(seed)
            outputs = rng.permutation(num_mappings)
        else:
            outputs = np.random.permutation(num_mappings)

        return Mapping(outputs, debug=debug)

    def get_output(self, index: int) -> int:
        return self.outputs[index]

    def get_input(self, output: int) -> int | None:
        return self.outputs_to_input[output]

    def __len__(self) -> int:
        return len(self.outputs)


class PrefixSuffixCorrespondence:
    def __init__(self, prefix_suffix_correspondence: list[tuple[int, int]], debug: bool = False):
        self.prefix_suffix_correspondence = prefix_suffix_correspondence

        self.prefix_suffix_to_index = {pair: i for i, pair in enumerate(prefix_suffix_correspondence)}

        self.max_prefix_index = max(pair[0] for pair in prefix_suffix_correspondence)
        self.max_suffix_indices = [max(pair[1] for pair in prefix_suffix_correspondence if pair[0] == i) for i in range(self.max_prefix_index + 1)]

    @classmethod
    def from_num_suffixes(cls, num_suffixes: list[int]) -> 'PrefixSuffixCorrespondence':
        prefix_suffix_correspondence = []
        for i in range(len(num_suffixes)):
            for j in range(num_suffixes[i]):
                prefix_suffix_correspondence.append((i, j))
        return cls(prefix_suffix_correspondence)

    def get_prefix_suffix(self, index: int) -> tuple[int, int]:
        return self.prefix_suffix_correspondence[index]

    def get_index(self, prefix_suffix: tuple[int, int]) -> int:
        return self.prefix_suffix_to_index[prefix_suffix]

    def __len__(self) -> int:
        return len(self.prefix_suffix_correspondence)

    def __iter__(self):
        return iter(self.prefix_suffix_correspondence)


class PrefixSuffixToIndexMapping:
    def __init__(self, mapping: Mapping, prefix_suffix_correspondence: PrefixSuffixCorrespondence):
        self.mapping = mapping
        self.prefix_suffix_correspondence = prefix_suffix_correspondence

        self.single_prefix_mappings = get_single_prefix_mappings(self.mapping, self.prefix_suffix_correspondence)

    def get_output(self, prefix_suffix: tuple[int, int]) -> int:
        return self.mapping.get_output(self.prefix_suffix_correspondence.get_index(prefix_suffix))

    def get_input(self, output: int) -> tuple[int, int]:
        return self.prefix_suffix_correspondence.get_prefix_suffix(self.mapping.get_input(output))

    def get_single_prefix_mapping(self, prefix_index: int) -> 'SinglePrefixMapping':
        return self.single_prefix_mappings[prefix_index]

    def __len__(self) -> int:
        return len(self.prefix_suffix_correspondence)


class SinglePrefixMapping:
    def __init__(self, mapping: list[int|None], debug: bool = False):
        self.mapping = mapping

        self.len = sum(1 for i in self.mapping if i is not None)

    def get_output(self, index: int) -> int | None:
        return self.mapping[index]

    def __len__(self) -> int:
        return self.len
