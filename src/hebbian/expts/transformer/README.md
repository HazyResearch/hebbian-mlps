# Transformer Experiments

This package contains the paper's Transformer-capacity experiments. They use
the associative-recall task `<junk_prefix> K <junk_suffix> Q V`, with the
model predicting the value associated with K at Q.

Entrypoints:

- `python -m hebbian.expts.transformer.run_num_facts`
- `python -m hebbian.expts.transformer.run_hidden_dim`
- `python -m hebbian.expts.transformer.summarize`
- `python -m hebbian.expts.transformer.plot`

Canonical schedule names:

- `attn_pretrain_then_insert`
- `insert_then_train_attn`

Paper presets include `full_num_facts_attn_pretrain`,
`paper_train99_num_facts`, `paper_evalacc100_hidden_dim`, and
`paper_trainacc100_hidden_dim`. Each run writes its resolved configuration,
raw binary-search artifacts, `capacity_points.csv`, and plots below its output
directory.
