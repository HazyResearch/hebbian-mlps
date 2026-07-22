# Hebbian MLP Constructions

This package contains the three bilinear Hebbian constructions used in the
paper. All use

\[
\phi(x) = (A_0 x) \odot (A_1 x).
\]

```python
from hebbian.methods.hebbian import HebbianConfig, HebbianMethod

method = HebbianMethod()
method.initialize(
    HebbianConfig(variant="whitened", m=128, ridge=1e-6),
    seed=42,
)
mlp, metrics = method.fit_or_construct(factset)
```

## Variants

| Variant | Features | Readout |
| --- | --- | --- |
| `unwhitened` | Random bilinear | Raw Hebbian outer product |
| `whitened` | Random bilinear | Full ridge readout |
| `data_dependent` | Fitted bilinear | Full ridge readout |

`m` defaults to `4 * d_model`. The historical experiment label
`cf_coord_whitened` maps to `variant="data_dependent"`; the label is retained
in result files so the paper plotters remain compatible.

## Layout

```text
hebbian/methods/hebbian/
|-- method.py        # Registry adapter and public configuration
|-- model.py         # BilinearFeatureMap and HebbianMLP
|-- construction.py  # Random and data-dependent feature construction
`-- readout.py       # Raw and full-ridge readouts
```

The standalone capacity experiments are the `fig_mlp_capacity_isotropic` and
`fig_mlp_capacity_anisotropic` targets in `scripts/paper/run.py`.
