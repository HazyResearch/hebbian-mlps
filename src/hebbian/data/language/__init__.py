"""Language data helpers for fact-editing experiments."""

from hebbian.data.language.authors import (
    AuthorFact,
    AuthorExampleDataset,
    AUTHOR_FACTS_CSV,
    author_collate_fn,
    augment_tokenizer_with_author_facts,
    build_author_example_groups,
    build_author_factset,
    default_author_facts_csv_path,
    flatten_author_example_groups,
    load_author_facts,
)

__all__ = [
    "AuthorFact",
    "AuthorExampleDataset",
    "AUTHOR_FACTS_CSV",
    "author_collate_fn",
    "augment_tokenizer_with_author_facts",
    "build_author_example_groups",
    "build_author_factset",
    "default_author_facts_csv_path",
    "flatten_author_example_groups",
    "load_author_facts",
]
