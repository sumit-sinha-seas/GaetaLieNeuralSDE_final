# Gaeta–Lie Neural SDEs: Symmetry-Regularized Learning of Stochastic Dynamics

Official code for the NeurIPS 2026 paper

> **Gaeta–Lie Neural SDEs: Symmetry-Regularized Learning of Stochastic Dynamics**
> Shida Liu\*, Sumit Sinha\*, L. Mahadevan
> *40th Conference on Neural Information Processing Systems (NeurIPS 2026)*
> \* Equal contribution.

Gaeta–Lie Neural SDEs improve drift and diffusion recovery of neural SDE surrogates using
**discovered approximate Itô SDE Lie symmetries**. They make no assumptions about the functional
form of the coefficients. The method has three stages:

1. **Stage I: train a surrogate.** Fit a flexible neural SDE `dX = f_θ(t,X) dt + Σ_θ(t,X) dW` by
   minimizing the Euler–Maruyama transition negative log-likelihood.
2. **Stage II: discover symmetries.** Freeze the surrogate and learn `m` projectable generators
   `X_a = Σ_r W_ar B_r` as linear combinations of a finite, editable library of canonical generator
   templates. The loss combines the Itô determining equations with Lie-closure, Gram-independence
   and norm terms. The Kozlov bound `d + 2` guides the choice of `m`. An optional held-out adequacy
   diagnostic selects the number of generators and abstains when no candidate passes.
3. **Stage III: refine the surrogate.** Freeze `W` and continue training the surrogate on
   `L_data + λ_GL · L_det`. The determining residual is evaluated on trajectory points and on
   collocation points sampled freely from the time–state domain.

> **Naming in the code.** The notebooks call Stage III **"Stage F"** (and sometimes call Stage I
> "Stage A" and Stage II "Stage B"). Some notebooks also use internal experiment IDs (`E1`, `E2`, …)
> in variable names and output file names. The table below maps each notebook to the paper.

## Repository structure

```
GaetaLieNeuralSDE/
├── experiments/
│   ├── 01_coefficient_recovery/   # 12 synthetic SDEs, d ∈ {1, 3, 20, 50}
│   ├── 02_ambient_collocation/    # 50D ambient collocation
│   ├── 03_controls/               # matched Sobolev control, stochastic SINDy
│   ├── 04_downstream/             # 64D MNIST score-based generation
│   ├── 05_ablations/              # scaling, generator count, collocation budget, loss ablations, ...
│   └── 06_visualizations/         # slice plots, roll-outs, OU generator alignment
├── requirements.txt
└── LICENSE
```

Each notebook runs on its own: it defines its SDEs, surrogate, basis library, discovery and
refinement. All settings are in the `CONFIG` dictionary in its first code cell.

## Experiment index

| Notebook | Paper reference | What it reproduces |
|---|---|---|
| `01_coefficient_recovery/gaeta_lie_1d.ipynb` | §5.1, Fig. 2, App. G, Tab. 12 | Ex 1.1–1.3 (`TRUE_SDE_NAME` = `ex_1_1` / `ex_1_2` / `ex_1_3`), R = 15, m = 3 |
| `01_coefficient_recovery/gaeta_lie_3d.ipynb` | §5.1, Fig. 2, App. G, Tab. 12 | Ex 3.1 `cyclic_reaction`, 3.2 `coupled_double_well`, 3.3 `tamed_lorenz`; R = 38, m = 5 |
| `01_coefficient_recovery/gaeta_lie_20d.ipynb` | §5.1, Fig. 2, App. G, Tab. 12 | Ex 20.1 `phase_difference`, 20.2 `tamed_lorenz`, 20.3 `cyclic_reaction`; R = 208, m = 5 |
| `01_coefficient_recovery/gaeta_lie_50d.ipynb` | §5.1–5.2, Fig. 2, Tab. 1, App. G–I | Ex 50.1 `ring_double_well_50d`, 50.2 `low_rank_ou_50d`, 50.3 `tanh_chain_50d`; R = 333, m = 50 |
| `02_ambient_collocation/ambient_collocation_50d.ipynb` | §5.3, Tab. 2, App. B.1 | Trajectory / jitter / box symmetry-batch compositions with 60 trajectories |
| `03_controls/sobolev_control_50d.ipynb` | §5.4, Tab. 4, App. B.2 | Gradient-matched Sobolev (state-Jacobian) penalty vs. Gaeta–Lie |
| `03_controls/stochastic_sindy_comparison.ipynb` | §5.4, Tab. 5, App. B.2 | Cross-validated stochastic SINDy vs. vanilla and Gaeta–Lie |
| `04_downstream/mnist64_score_diffusion.ipynb` | §5.5, App. F, Tab. 11 | VP score-based diffusion in 64D PCA space with reverse-drift regularization |
| `05_ablations/generator_count_sweep_50d.ipynb` | §5.7, App. K, Fig. 9 | MSE reduction vs. number of generators m |
| `05_ablations/computational_scaling_50d.ipynb` | §5.6, App. M, Fig. 11 | Stage-II discovery time and Stage-III per-step overhead vs. m |
| `05_ablations/dimension_scaling.ipynb` | §5.6, App. M, Tab. 15 | Residual scaling and λ_GL calibration for d ∈ {1, 3, 20, 50} |
| `05_ablations/collocation_budget_50d.ipynb` | App. N | Extra Stage-III collocation points ∈ {0, 128, 256, 512, 1024} |
| `05_ablations/ablations_20d.ipynb` | §5.6–5.7, App. N.1, Tab. 17–19 | Ex 20.2 accuracy–collocation–runtime tradeoff; loss-weight and zero-weight ablations |
| `05_ablations/alternating_discovery_refinement_50d.ipynb` | §5.7, App. N.1, Tab. 20 | Repeated discovery–refinement cycles and Δ_k |
| `05_ablations/library_adequacy_20d.ipynb` | §4.2, App. C.5 | Held-out adequacy diagnostic: R = 208 library vs. R = 42 control |
| `06_visualizations/ex1_1_slices_and_rollouts.ipynb` | App. J–L, Fig. 6–8 | Drift/diffusion slices, paired roll-outs, learned-generator summary |
| `06_visualizations/ou_generator_alignment.ipynb` | §5.7, App. K, Fig. 10 | Learned vs. analytic generators on a 1D OU SDE |

## Installation

The code uses [JAX](https://github.com/google/jax). We recommend a GPU for the 20D, 50D and MNIST
experiments; the 1D and 3D notebooks run on a CPU.

```bash
git clone https://github.com/sumit-sinha-seas/GaetaLieNeuralSDE_final.git
cd GaetaLieNeuralSDE_final
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For an NVIDIA GPU, install the matching JAX build instead, e.g.:
# pip install -U "jax[cuda12]"
```

Then open a notebook and run the cells from top to bottom:

```bash
jupyter lab experiments/01_coefficient_recovery/gaeta_lie_1d.ipynb
```

The notebooks also run unchanged on Google Colab. Outputs (CSV, JSON, figures) go to the current
working directory.

## Hyperparameters

The paper's default settings (Appendix B):

| | 1D | 3D | 20D | 50D |
|---|---|---|---|---|
| Surrogate MLP (width × depth, tanh) | 64 × 3 | 96 × 3 | 96 × 3 | 192 × 3 |
| Stage I (Adam, batch 512) | lr 2e-3, 1500 ep | lr 2e-3, 2000 ep | lr 2e-3, 2000 ep | lr 2e-3, 2000 ep |
| Library size R / generators m | 15 / 3 | 38 / 5 | 208 / 5 | 333 / 50 |
| Stage II lr | 1e-2 | 1e-2 | 1e-2 | 5e-3 |
| (λ_det, λ_close, λ_ind, λ_norm) | (1, 0.1, 0.05, 0.01) | same | same | (1, 0.5, 0.05, 0.01) |
| Stage III lr / epochs | 1e-3 / 1200 | 1e-3 / 1200 | 1e-3 / 1200 | 3e-4 / 800 |
| λ_GL | 0.05 | 0.05 | 0.05 | 1.05 × 10⁵ |

Each notebook ships with the configuration of its last run. Before reproducing a specific table
entry, check `N_TRAJECTORIES`, `SEED` and `TRUE_SDE_NAME` against the paper. For example, the 50D
data-scarce regime uses 60 trajectories.

## Citation

```bibtex
@inproceedings{liu2026gaetalie,
  title     = {Gaeta--Lie Neural {SDE}s: Symmetry-Regularized Learning of Stochastic Dynamics},
  author    = {Liu, Shida and Sinha, Sumit and Mahadevan, L.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

The symmetry framework builds on the Itô SDE Lie-point symmetry theory of Gaeta and Quintero (1999),
Gaeta (2017) and Kozlov (2010). The PhysioNet and Hopper experiments follow the setup of Stable
Neural SDEs (Oh et al., ICLR 2024).

## License

Released under the MIT License. See [LICENSE](LICENSE).
