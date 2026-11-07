# Gaeta–Lie Neural SDEs: Symmetry-Regularized Learning of Stochastic Dynamics

Code for the NeurIPS 2026 paper.

Each experiment is a self-contained Jupyter notebook. Run its cells from top to bottom. All settings
are in the `CONFIG` dictionary in the first code cell.

In the code, the pipeline stages are named as follows:

- **Stage I**: train the neural SDE surrogate.
- **Stage II**: discover symmetry generators from the basis library.
- **Stage F**: refine the surrogate with the symmetry penalty (Stage III in the paper).

## Setup

```bash
git clone https://github.com/sumit-sinha-seas/GaetaLieNeuralSDE_final.git
cd GaetaLieNeuralSDE_final
pip install -r requirements.txt
jupyter lab
```

For GPU support, install the matching JAX build (e.g. `pip install -U "jax[cuda12]"`). A GPU is
recommended for the 20D, 50D and MNIST notebooks.

## Notebooks

| Notebook | Paper |
|---|---|
| **01_coefficient_recovery/** | |
| `gaeta_lie_1d.ipynb`: Ex 1.1–1.3 | §5.1, Table 12 |
| `gaeta_lie_3d.ipynb`: Ex 3.1–3.3 | §5.1, Table 12 |
| `gaeta_lie_20d.ipynb`: Ex 20.1–20.3 | §5.1, Table 12 |
| `gaeta_lie_50d.ipynb`: Ex 50.1–50.3 | §5.1–5.2, Tables 1, 12–14 |
| **02_ambient_collocation/** | |
| `ambient_collocation_50d.ipynb`: trajectory / jitter / box collocation | §5.3, Table 2 |
| **03_controls/** | |
| `sobolev_control_50d.ipynb`: matched Sobolev penalty | §5.4, Table 4 |
| `stochastic_sindy_comparison.ipynb`: stochastic SINDy | §5.4, Table 5 |
| **04_downstream/** | |
| `mnist64_score_diffusion.ipynb`: score-based generation in 64D | §5.5, Table 11 |
| **05_ablations/** | |
| `generator_count_sweep_50d.ipynb`: number of generators | Fig. 9 |
| `computational_scaling_50d.ipynb`: runtime cost | Fig. 11 |
| `dimension_scaling.ipynb`: residual scaling and λ_GL across dimensions | Table 15 |
| `collocation_budget_50d.ipynb`: extra collocation points | App. N |
| `ablations_20d.ipynb`: collocation–runtime tradeoff, loss-weight ablations | Tables 17–19 |
| `alternating_discovery_refinement_50d.ipynb`: repeated discovery–refinement cycles | Table 20 |
| `library_adequacy_20d.ipynb`: held-out library-adequacy diagnostic | App. C.5 |
| **06_visualizations/** | |
| `ex1_1_slices_and_rollouts.ipynb`: drift/diffusion slices and roll-outs | Figs. 6–8 |
| `ou_generator_alignment.ipynb`: learned vs. analytic OU generators | Fig. 10 |

To choose an example SDE, set `CONFIG["TRUE_SDE_NAME"]` (the available names are listed in each
notebook). Outputs are written to the current working directory.

## License

MIT

## Citation

```bibtex
@inproceedings{liu2026gaetalie,
  title     = {Gaeta--Lie Neural {SDE}s: Symmetry-Regularized Learning of Stochastic Dynamics},
  author    = {Liu, Shida and Sinha, Sumit and Mahadevan, L.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
