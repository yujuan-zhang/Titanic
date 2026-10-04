# Titanic
This project will predict the survival and death rates of passengers on the Titanic.


## Run the included example

This repository predicts the binary `Survived` label for Kaggle Titanic
passengers. Start with `titanic_advanced.ipynb`, which loads the bundled CSVs,
creates features, trains a stacking ensemble and exports predictions/plots.
`quick_improvements.py` is a feature-engineering example; it does not train the
full model or create a submission by itself.

From a terminal:

```bash
git clone https://github.com/yujuan-zhang/Titanic.git
cd Titanic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyterlab
python -m jupyterlab
```

Open `titanic_advanced.ipynb` and run cells from top to bottom with the notebook's
working directory set to the repository root. These commands install the
notebook's declared imports; an exact dependency lockfile is not provided.

## Input and output

- `train.csv`: one passenger per row, with `PassengerId`, `Pclass`, `Name`,
  `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`, and
  reference label `Survived` (0 or 1).
- `test.csv`: the same passenger fields without `Survived`.
- `submission_advanced.csv`: `PassengerId,Survived`, one prediction per test row.
- Plots: `feature_distributions.png`, `enhanced_correlation.png`,
  `model_performance.png`, and `stacking_importance.png` in the working directory.

Running the notebook overwrites these output files. Verify the submission has
the same row count and passenger IDs as `test.csv`, no missing predictions and
only labels 0/1. The committed submission is a reference artifact, not a promise
of identical predictions across dependency versions.

All code cells were executed in order on the bundled data in a Linux Docker
container limited to 2 CPUs and 4 GB RAM. Execution took about 47 seconds,
excluding dependency installation, and produced 418 predictions plus all four
plots. Tested versions: Python 3.11.16, pandas 2.0.3, NumPy 1.26.4,
matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.7.2, XGBoost 3.2.0 and
LightGBM 4.7.0. Runtime varies with hardware and versions. Training uses
five folds and several tree ensembles. Accuracy-improvement claims in the
helper script are not guarantees. Its survival-rate features are computed from
training labels; fit such features within training folds before using them for
validation scores.
