# Fact Editing

These experiments test whether a Transformer's fact-storing MLP can be edited
or replaced while the rest of the model remains fixed. The paper compares
MLP swapping with MEMIT, AlphaEdit, and ROME on an author-relation task.

## Reproduce The Paper Results

Run the h=512 paper preset and generate the editing-score plot:

```bash
python scripts/paper/run.py fig_fact_editing_score --mode run-and-plot
```

The same run produces the data used by the appendix plot and table:

```bash
python scripts/paper/run.py fig_fact_editing_nonfact_ppl_ratio --mode plot-only
python scripts/paper/run.py tab_fact_editing_h512 --mode plot-only
```

Generated checkpoints and results are written below
`artifacts/paper/results/fact_editing/`. They are not included in git.

## Workflow

The maintained Python modules separate the experiment into four stages:

1. `hebbian.expts.fact_editing.train_base` trains the one-layer Transformer
   and writes its checkpoint, embeddings, and metadata.
2. `hebbian.expts.fact_editing.run_edit` evaluates one editing method.
3. `hebbian.expts.fact_editing.pipeline` launches the paper method and
   edit-fraction grid across the requested GPUs.
4. `hebbian.expts.fact_editing.summarize_results` selects the best result for
   each method and edit count.

`plot_edit_fraction_sweep` generates the score and non-fact-perplexity plots,
while `export_paper_table` writes the appendix table. The packaged author data
lives at `src/hebbian/data/language/book_authors.csv`.
