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

## Hyperparameters

The analyses use the final hyperparameters saved with the model in `02_models_and_support/primary_row_split_ebm.pkl`.
These came from the original parallel Optuna search. A parallel search does not return identical settings on each run, so the search is not part of this package and the settings are fixed.

## Shape-function resolution

The percentages of each variable's domain resolved in Section 4 are the shares of the 50 common-grid points where the 95% envelope excludes zero, from `stability_main_effect_grid.csv`.
