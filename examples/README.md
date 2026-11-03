# DTCD Examples

These examples introduce the main capabilities of the DTCD (Datasets and Toolkit for Causal Discovery) API, one step at a time.

## Prerequisites
To run these examples, ensure DTCD is installed:
```bash
pip install -e .
```

All eight run on the core install. Examples 04 and 05 additionally run PCMCI and the PC algorithm when the optional `tigramite` and `causal-learn` packages are installed (`pip install -e ".[discovery]"`); without them they only perform the export.

## Example Scripts

| Script | Description | Runtime |
|--------|-------------|---------|
| `01_basic_generation.py` | Generates a simple multivariate time series dataset with default settings. | ~1s |
| `02_parameter_sweep.py` | Sweeps across sparsity levels and noise types to demonstrate correlation effects. | ~2s |
| `03_missingness_variants.py` | Applies MCAR and MNAR missingness masks to a generated trajectory. | ~1s |
| `04_export_to_tigramite.py` | Exports ground-truth links to Tigramite format and optionally runs PCMCI. | ~1s |
| `05_export_to_causallearn.py` | Exports links to causal-learn matrix format and optionally runs the PC algorithm. | ~1s |
| `06_counterfactual_twin.py` | Evaluates a hard "do-intervention" by generating an aligned counterfactual twin. | ~1s |
| `07_spatiotemporal.py` | Generates data across a 2D spatial grid and applies a hard intervention to show diffusion. | ~3s |
| `08_full_pipeline_minimal.py` | End-to-end journey from generation to inspection to export. | ~1s |

## Further Reading
- [docs/MANUAL.md](../docs/MANUAL.md) covers the API in more depth, including how to add new mechanisms, noise models, missingness processes, and baselines.
- The repository [README](../README.md) explains how to reproduce the paper's benchmark.
