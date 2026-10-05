# Titanic

## What it does

Build a binary classifier that predicts whether an individual Titanic
passenger survived, using passenger information such as ticket class, sex,
age, family size and fare. The target is `Survived`: 0 for a passenger who died, 1 for a passenger who survived.

The main notebook follows the complete example:

```text
Passenger CSVs → inspect missing values → prepare features
               → train first-level models → combine their predictions
               → predict the test passengers → save submission and plots
```

It combines several classifiers through stacking and a weighted XGBoost/
LightGBM prediction. The notebook fits models during the run; no pretrained
weights need to be downloaded. Its outputs let you inspect the data, compare
models and prepare a Kaggle-style submission for the 418 test passengers.

`notebooks/titanic_advanced.ipynb` is the local example. The other notebook
preserves the original Kaggle version. `quick_improvements.py` demonstrates
feature-engineering ideas only; it does not run the full classifier or create
a submission by itself. Evaluation and label-derived feature cautions are
kept in the notes below.

## Input

- `data/train.csv`: one passenger per row, with `PassengerId`, `Pclass`, `Name`,
  `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`, and
  reference label `Survived` (0 or 1).
- `data/test.csv`: the same passenger fields without `Survived`.

## Output

- `outputs/submission_advanced.csv`: `PassengerId,Survived`, one prediction per test row.
- Plots: `feature_distributions.png`, `enhanced_correlation.png`,
  `model_performance.png`, and `stacking_importance.png` in `outputs/`.

## Try it

### Run the included example

This repository predicts the binary `Survived` label for Kaggle Titanic
passengers. Start with `notebooks/titanic_advanced.ipynb`, which loads the bundled CSVs,
creates features, trains a stacking ensemble and exports predictions/plots.
`quick_improvements.py` is a feature-engineering example; it does not train the
full model or create a submission by itself.

Use an installed Python 3.11 interpreter for the checked training-package versions.
From a terminal:

```bash
git clone https://github.com/yujuan-zhang/Titanic.git
cd Titanic
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `notebooks/titanic_advanced.ipynb` and run cells from top to bottom.
The local data paths work with the kernel directory set to either the project
root or `notebooks/`. Results go to the project's `outputs/` directory. `requirements.txt` pins the numerical and training libraries to the checked
versions; JupyterLab has a version range. Transitive dependencies are not fully
locked, and this exact requirements file has not been freshly installed.

Running the notebook overwrites these output files. Verify the submission has
the same row count and passenger IDs as `data/test.csv`, no missing predictions and
only labels 0/1. Run the standard-library output check from any directory:

```bash
python verify_outputs.py
```

It checks the 418 bundled test IDs in order, binary predictions, and the four
nonempty PNG outputs. Missing or malformed outputs cause a nonzero exit code.
It checks the existing files; it does not train the model. The committed submission is a reference artifact, not a promise
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

### Project layout

```text
Titanic/
├── README.md
├── requirements.txt
├── quick_improvements.py
├── verify_outputs.py
├── data/                # Bundled training and test passengers
├── notebooks/           # Main local example and original Kaggle reference
└── outputs/             # Submission and four generated plots
```

`notebooks/titanic_advanced.ipynb` is the checked local example.
`notebooks/see-it-stack-it-ensemble-it-86-precision.ipynb` is the original
Kaggle reference and retains its `/kaggle/input/titanic/` input paths; it is
not the local quick-start. Its computations and saved notebook content are
preserved. The folder cleanup does not change model fitting or predictions.

Run `python quick_improvements.py` from the project root for the standalone
feature-engineering example. Both Python entry points locate their files
relative to the script, so they also work when launched by absolute path.

