# Evidence-supported design space of fly-ash pervious concrete: data and code

Processed dataset and analysis code for the paper. All random seeds are fixed.

## Setup

Python 3.12 with the exact versions in `requirements.txt`:

    pip install -r requirements.txt

The versions matter: the optimiser results depend on them, and other versions can give slightly different numbers.
A version warning from scikit-learn when the saved support models load is expected.

## Running

    python run_all.py                    # everything, then the comparison with the paper
    python run_all.py --validation-only  # validation only, about 15-25 minutes
    python run_all.py --sensitivity-only # sensitivity and scenario analyses only, a few minutes
    python check_results.py              # compare existing results with the paper

The full run takes several hours on a desktop computer. The design-space sweep and differential evolution take the longest.

## Contents

| Folder | Content |
|---|---|
| `01_core_data` | `analysis_records.csv` is the processed dataset of 505 observations from 22 studies; `phase5_augmented_dataset.csv` is the model input table; `porosity_by_row.csv` holds compiled porosity |
| `02_models_and_support` | Saved final EBM (16 outer bags), support models and local residual model |
| `validation` | Repeated CV, permutation and leverage tests, leave-one-study-out validation, Figures 3 and S4 |
| `flyash analysis` | Resampled shape functions and +10 percentage-point FA contrasts |
| `08_analysis_code_suite` | Differential evolution over 20 seeds and 37 targets |
| `robustness_verification` | Sobol screening, threshold sensitivity and final reconciliation |
| `sensitivity_and_scenarios` | Neighbouring observations and studies for Table 4, KDE bandwidth and Isolation Forest settings, safety-margin multiplier, fly ash allocation and transport scenarios, Figure 9 |
| `results` | Stored outputs: `validation`, `robustness` (threshold-grid results used by the sensitivity analyses) and `sensitivity` |

## Hyperparameters

The analyses use the final hyperparameters saved with the model in `02_models_and_support/primary_row_split_ebm.pkl`.
These came from the original parallel Optuna search. A parallel search does not return identical settings on each run, so the search is not part of this package and the settings are fixed.

## Shape-function resolution

The percentages of each variable's domain resolved in Section 4 are the shares of the 50 common-grid points where the 95% envelope excludes zero, from `stability_main_effect_grid.csv`.

## Tables and figures

| Item in the paper | Script | Output |
|---|---|---|
| Table 3, Figures 3 and S4, leave-one-study-out validation (Section 4.2) | `validation/run_validation.py` | `results/validation/` |
| Table 4, observations and studies near each candidate | `sensitivity_and_scenarios/table4_neighbour_studies.py` | `results/sensitivity/table4_neighbour_studies.csv` |
| Table S4, KDE bandwidth and Isolation Forest contamination | `sensitivity_and_scenarios/support_settings.py` | `results/sensitivity/support_settings.csv` |
| Table S5, safety-margin multiplier k | `sensitivity_and_scenarios/margin_multiplier.py` | `results/sensitivity/margin_multiplier.csv` |
| Fly ash allocation and transport scenarios (Section 4.4.2) | `sensitivity_and_scenarios/carbon_scenarios.py` | `results/sensitivity/carbon_scenarios.csv` |
| Figure 9 | `sensitivity_and_scenarios/figure_9.py` | `results/sensitivity/figure_9.png`, `figure_9_data.csv` |

The sensitivity and scenario analyses use the primary Sobol sweep (2^17 points, seed 20260829) and check their
baseline against the threshold-grid results before writing anything. They read those results from
`robustness_verification/outputs` after a full run, and from the stored copy in `results/robustness` otherwise.

## Licence

MIT; see `LICENSE`.

## Citation

See `CITATION.cff`.
