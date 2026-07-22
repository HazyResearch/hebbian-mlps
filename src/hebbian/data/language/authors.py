"""Author-fact language dataset helpers.

This module provides the author task used by the fact-editing experiments.
The task is still:
    book -> author
wrapped in natural-language paraphrases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from hebbian.data.synthetics.factsets import Factset, create_identity_mapping


AUTHOR_FACTS_CSV = "book_authors.csv"

AUTHOR_SENTENCE_TEMPLATES = [
    "The author of {book} is {author}",
    "Who is the author of {book}? It's {author}",
    "Do you know the author of {book}? It's {author}",
    "As for the author of {book}, that would be {author}",
    "The one credited as author of {book} is {author}",
    "If you're wondering who the author of {book} is, it's {author}",
    "Well, {book}'s author is {author}",
    "The creator and author of {book} is {author}",
    "When it comes to {book}, the author is {author}",
    "Everyone knows the author of {book} is {author}",
    "Let's not forget the author of {book}; it's {author}",
    "Guess who the author of {book} is? It's {author}",
    "By the way, the author of {book} is {author}",
    "Who wrote {book}? The author is {author}",
    "One name as the author of {book} stands out: {author}",
    "Looking at {book}, the author is clearly {author}",
    "The person known as the author of {book} is {author}",
    "The literary mind behind {book} is the author {author}",
    "The name behind the author credit for {book} is {author}",
    "People recognize the author of {book} as {author}",
    "In the case of {book}, the author turns out to be {author}",
    "Award-winning author of {book}? That would be {author}",
    "Scholars agree: the author of {book} is {author}",
    "The publication of {book} lists, as its author, {author}",
    "It's confirmed that the author of {book} is {author}",
    "The individual who was author of {book} is {author}",
    "The person who wrote {book} is the author {author}",
    "The person responsible as author of {book} is {author}",
    "The author role for {book} was filled by {author}",
    "The book {book} was written by the author {author}",
    "Fans of {book} know the author is {author}",
    "The author of {book} goes to {author}",
]


@dataclass(frozen=True)
class AuthorFact:
    book: str
    author: str


def default_author_facts_csv_path() -> str:
    """Return the packaged author-facts CSV path."""
    return str(resources.files(__package__).joinpath(AUTHOR_FACTS_CSV))


def load_author_facts(csv_path: str, num_facts: int | None = None) -> List[AuthorFact]:
    """Load author facts from a CSV."""
    df = pd.read_csv(csv_path).dropna()
    if num_facts is not None:
        df = df.iloc[:num_facts]
    facts = [AuthorFact(book=row["book"], author=row["author"]) for _, row in df.iterrows()]
    if len({fact.book for fact in facts}) != len(facts):
        raise ValueError("Expected unique books in author facts.")
    if len({fact.author for fact in facts}) != len(facts):
        raise ValueError("Expected unique authors in author facts.")
    return facts


def author_facts_to_dicts(facts: Sequence[AuthorFact]) -> List[Dict[str, str]]:
    return [asdict(fact) for fact in facts]


def author_facts_from_dicts(items: Sequence[Dict[str, str]]) -> List[AuthorFact]:
    return [AuthorFact(**item) for item in items]


def augment_tokenizer_with_author_facts(
    tokenizer: PreTrainedTokenizerBase,
    facts: Sequence[AuthorFact],
) -> List[str]:
    """Add book and author strings as atomic tokens, matching old NTP behavior."""
    added_tokens = sorted(
        {
            f" {fact.book}"
            for fact in facts
        }
        | {
            f" {fact.author}"
            for fact in facts
        }
    )
    tokenizer.add_tokens(added_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return added_tokens


def _single_token_id(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected a single token after tokenizer augmentation for {text!r}, got {token_ids}"
        )
    return token_ids[0]


def get_author_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    facts: Sequence[AuthorFact],
) -> Tuple[List[int], List[int]]:
    book_token_ids = [_single_token_id(tokenizer, f" {fact.book}") for fact in facts]
    author_token_ids = [_single_token_id(tokenizer, f" {fact.author}") for fact in facts]
    if len(set(book_token_ids)) != len(book_token_ids):
        raise ValueError("Book token ids must be unique for the fact-editing task.")
    if len(set(author_token_ids)) != len(author_token_ids):
        raise ValueError("Author token ids must be unique for the fact-editing task.")
    return book_token_ids, author_token_ids


def build_author_sentences(
    facts: Sequence[AuthorFact],
    num_rephrases: int,
) -> List[List[str]]:
    if num_rephrases > len(AUTHOR_SENTENCE_TEMPLATES):
        raise ValueError(
            f"Requested {num_rephrases} rephrases but only {len(AUTHOR_SENTENCE_TEMPLATES)} are available."
        )
    groups: List[List[str]] = []
    for fact in facts:
        groups.append(
            [
                template.format(book=fact.book, author=fact.author)
                for template in AUTHOR_SENTENCE_TEMPLATES[:num_rephrases]
            ]
        )
    return groups


def build_author_example_groups(
    tokenizer: PreTrainedTokenizerBase,
    facts: Sequence[AuthorFact],
    num_rephrases: int,
    *,
    train_last_token: bool = False,
    fail_on_token_miss: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Build grouped tokenized examples for the book->author language task."""
    groups: List[List[Dict[str, Any]]] = []
    sentences = build_author_sentences(facts, num_rephrases=num_rephrases)
    for fact_index, (fact, fact_sentences) in enumerate(zip(facts, sentences)):
        value_token_ids = tokenizer.encode(f" {fact.author}", add_special_tokens=False)
        if not value_token_ids:
            raise ValueError(f"Empty tokenization for author {fact.author!r}")
        group: List[Dict[str, Any]] = []
        for sentence in fact_sentences:
            tokenized = tokenizer(
                sentence,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if fail_on_token_miss and tokenized[-len(value_token_ids):] != value_token_ids:
                raise ValueError(
                    f"Value tokens were not the final tokens for sentence {sentence!r}: "
                    f"expected suffix {value_token_ids}, got {tokenized}"
                )
            if tokenized[-len(value_token_ids):] != value_token_ids:
                raise ValueError(
                    f"Sentence {sentence!r} does not end with author tokens {value_token_ids}."
                )

            input_ids = list(tokenized[:-1])
            labels = list(tokenized[1:])
            labels_last = list(labels)
            labels_all_but_last = list(labels)
            labels_last[:-len(value_token_ids)] = [-100] * max(0, len(labels_last) - len(value_token_ids))
            labels_all_but_last[-len(value_token_ids):] = [-100] * len(value_token_ids)
            train_labels = labels_last if train_last_token else labels
            group.append(
                {
                    "fact_index": fact_index,
                    "text": sentence,
                    "input_ids": input_ids,
                    "labels": train_labels,
                    "labels_full": labels,
                    "labels_last": labels_last,
                    "labels_all_but_last": labels_all_but_last,
                    "value_token_ids": list(value_token_ids),
                }
            )
        groups.append(group)
    return groups


def flatten_author_example_groups(
    example_groups: Sequence[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    flat_examples: List[Dict[str, Any]] = []
    for group in example_groups:
        flat_examples.extend(group)
    return flat_examples


class AuthorExampleDataset(Dataset):
    """Dataset wrapper over flat author examples."""

    def __init__(self, examples: Sequence[Dict[str, Any]], label_key: str = "labels") -> None:
        self.examples = list(examples)
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        return {
            "input_ids": example["input_ids"],
            "labels": example[self.label_key],
        }


def author_collate_fn(features: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_tensors = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
    label_tensors = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
    batch_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=-100)
    batch_labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    return batch_inputs, batch_labels


def build_author_factset(
    tokenizer: PreTrainedTokenizerBase,
    facts: Sequence[AuthorFact],
    full_input_embeddings: torch.Tensor,
    full_output_embeddings: torch.Tensor,
    *,
    factset_dtype: torch.dtype | None = None,
    fact_input_embedding_mode: str = "token",
    book_component_token_ids: Sequence[Sequence[int]] | None = None,
) -> tuple[Factset, Dict[str, Any]]:
    """Build the reduced fact-level embeddings used by the inserted MLP."""
    book_token_ids, author_token_ids = get_author_token_ids(tokenizer, facts)
    if fact_input_embedding_mode == "token":
        fact_input_embeddings = full_input_embeddings[book_token_ids].clone()
    elif fact_input_embedding_mode == "normalized_token":
        fact_input_embeddings = F.normalize(
            full_input_embeddings[book_token_ids].clone(),
            dim=1,
        )
    elif fact_input_embedding_mode == "average_compound_normalized":
        if book_component_token_ids is None:
            raise ValueError(
                "book_component_token_ids is required for average_compound_normalized fact input embeddings."
            )
        if len(book_component_token_ids) != len(facts):
            raise ValueError(
                "book_component_token_ids must contain one token-id sequence per fact."
            )
        averaged_inputs = []
        for fact, component_ids in zip(facts, book_component_token_ids):
            if not component_ids:
                raise ValueError(f"Empty component tokenization for book {fact.book!r}.")
            component_tensor = torch.tensor(
                component_ids,
                dtype=torch.long,
                device=full_input_embeddings.device,
            )
            averaged_inputs.append(full_input_embeddings.index_select(0, component_tensor).mean(dim=0))
        fact_input_embeddings = F.normalize(torch.stack(averaged_inputs, dim=0), dim=1)
    else:
        raise ValueError(
            "Unknown fact_input_embedding_mode "
            f"{fact_input_embedding_mode!r}; expected 'token', 'normalized_token', or "
            "'average_compound_normalized'."
        )
    fact_output_embeddings = full_output_embeddings[author_token_ids].clone()
    if factset_dtype is not None:
        fact_input_embeddings = fact_input_embeddings.to(dtype=factset_dtype)
        fact_output_embeddings = fact_output_embeddings.to(dtype=factset_dtype)
    factset = Factset(
        input_embeddings=fact_input_embeddings,
        output_embeddings=fact_output_embeddings,
        mapping=create_identity_mapping(len(facts)),
        d_model=fact_input_embeddings.shape[1],
        vocab_size=len(facts),
    )
    return factset, {
        "book_token_ids": book_token_ids,
        "author_token_ids": author_token_ids,
        "fact_input_embedding_mode": fact_input_embedding_mode,
        "book_component_token_ids": (
            [list(ids) for ids in book_component_token_ids]
            if book_component_token_ids is not None
            else None
        ),
    }
