# Classifier-Dependent Kernel Selection in KPCA-Based Network Intrusion Detection

A Python/Spyder project that evaluates how Kernel Principal Component Analysis (KPCA) kernel choices affect the performance of a selected machine-learning classifier on network traffic associated with the CIC-IDS2017 dataset.

The current implementation is a **memory-aware, sampled experiment**. It does not run exact KPCA on the full combined dataset. The code is configured to run one classifier per execution and test that classifier across five kernels and three train-test split settings.

## 1. Dataset files expected by the code

The script expects these three CSV files, by these exact names:

1. `Tuesday-WorkingHours.pcap_ISCX.csv`
2. `Wednesday-workingHours.pcap_ISCX.csv`
3. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

By default, `DATA_DIR = ""`, so place the Python script and all three CSV files in the same folder. Alternatively, set `DATA_DIR` in the script to the directory containing the CSVs, for example:

```python
DATA_DIR = r"D:\ML_Project\datasets"
```

The filenames are associated with CICIDS2017. The script verifies that the files exist and that the combined data contains a `Label` column. It prints the number of combined rows after rows with missing labels are removed, the number of feature columns, and the number of distinct labels. Exact dataset statistics should be taken from the output of the final run with the exact files used for that run.

## 2. Experimental configuration in the current script

| Setting | Current code value |
|---|---|
| Classifiers supported | XGBoost, CatBoost, LightGBM, KNN, AdaBoost |
| Classifier selected by default | `AdaBoost` |
| KPCA kernels | Linear, polynomial, RBF, sigmoid, cosine |
| Test sizes | 0.2, 0.4, 0.6 |
| Equivalent train:test ratios | 80:20, 60:40, 40:60 |
| Maximum sampled rows | 10,000 (`MAX_ROWS`) |
| KPCA fitting reference rows | Up to 1,200 training rows (`KPCA_FIT_ROWS`) |
| KPCA components | 2 (`N_COMPONENTS`) |
| Random state | 0 |
| Results file | `kpca_selected_algorithm_results.csv` |
| Expected configurations per successful run | 1 classifier × 5 kernels × 3 test sizes = 15 |

Across all five classifiers, the planned matrix contains 75 configurations. However, the current script runs **only the classifier named by `SELECTED_ALGORITHM` each time**. A run may contain fewer than 15 results if some configurations fail or a split is skipped.

## 3. Data preparation and preprocessing

The script performs these steps:

1. Reads the three CSV files and concatenates them row-wise.
2. Strips whitespace from column names and removes duplicate column names.
3. Removes rows where `Label` is missing and strips whitespace from label values.
4. Takes an approximately proportional, class-wise sample using `MAX_ROWS = 10000` when the combined data has more than 10,000 rows. If the data has 10,000 rows or fewer, it uses all available rows after the missing-label cleanup.
5. Uses `Label` as the target and converts the remaining columns to numeric values; values that cannot be converted, as well as positive/negative infinity, become missing values.
6. Drops feature columns that contain no non-missing values.
7. Encodes target labels using `LabelEncoder`.
8. Removes sampled rows belonging to classes with fewer than two examples, because stratified train-test splitting requires sufficient examples per class.
9. For each test size, fits median imputation and standard scaling on the training split only, then applies those fitted transformations to the test split.
10. Fits KPCA on a random subset of up to 1,200 training rows, then transforms the full training and test splits using that fitted KPCA model.

The script does **not** explicitly remove duplicate data rows. It also does not perform a separate train/test deduplication step. Record these details accurately when describing the method.

## 4. Kernel PCA configuration

The code uses `sklearn.decomposition.KernelPCA` with `n_components=2`, `eigen_solver="arpack"`, `n_jobs=1`, and random state 0. Kernel settings are:

| Kernel label in results | scikit-learn kernel | Parameters |
|---|---|---|
| `linear` | `linear` | None specified |
| `polynomial` | `poly` | `degree=3` |
| `rbf` | `rbf` | `gamma=1.0` |
| `sigmoid` | `sigmoid` | `gamma=1.0` |
| `cosine` | `cosine` | None specified |

Because KPCA is fitted on at most 1,200 training reference rows and the data may be sampled to at most 10,000 rows, the resulting experiments are **sampled/reference-subset KPCA experiments**, not full-dataset exact-KPCA experiments. Report these settings in any paper, presentation, or results description.

## 5. Classifiers and parameters

Only the classifier selected in `SELECTED_ALGORITHM` is instantiated for a run. The available choices and main parameters in the current script are:

- **XGBoost:** 100 estimators, maximum depth 6, learning rate 0.1, subsample 0.8, column subsample 0.8, histogram tree method.
- **CatBoost:** 100 iterations, depth 6, learning rate 0.1.
- **LightGBM:** 100 estimators, learning rate 0.1.
- **KNN:** 5 neighbors.
- **AdaBoost:** 100 estimators.

The random state is 0 where supported by the implementation. These are the parameters in the current script; do not describe them as hyperparameter-optimized unless a separate tuning procedure is run.

## 6. Evaluation metrics

For each successful configuration, the script records:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Classifier training time in seconds
- Prediction time in seconds
- Number of rows used
- Number of KPCA fitting rows
- Number of KPCA components

Precision, recall, and F1-score are calculated with `average="weighted"` and `zero_division=0`. Training time measures the classifier's `.fit()` call; prediction time measures the classifier's `.predict()` call. The current timers do not include KPCA fitting/transformation time in those two fields.

## 7. How to run in Spyder

### Install dependencies

In the IPython console associated with the same Python environment used by Spyder, run:

```python
%pip install xgboost lightgbm catboost scikit-learn pandas numpy
```

Restart the console/kernel if required after installation.

### Configure and execute

1. Place the script and the three CSV files in one folder, or set `DATA_DIR` to the CSV folder.
2. Open `kpca_nids_spyder_one_algorithm.py` in Spyder (or the provided renamed copy of this script).
3. Set `SELECTED_ALGORITHM` to exactly one of `"XGBoost"`, `"CatBoost"`, `"LightGBM"`, `"KNN"`, or `"AdaBoost"`.
4. Run the script.
5. Check the console for the combined-row count, feature count, label count, sampled label counts, configuration results, and any failed configurations.
6. Inspect `kpca_selected_algorithm_results.csv` in the working directory.
7. Repeat for the other classifiers.

### Preserve results from each classifier

The current code writes to the same filename, `kpca_selected_algorithm_results.csv`, on each run. Rename or copy the output after each successful run (for example, `results_AdaBoost.csv`, `results_XGBoost.csv`) before running another classifier, or update `OUTPUT_CSV` before each run. Otherwise, a later run can overwrite the previous classifier's results. Combine the five saved result files to build the 75-configuration comparison, accounting for any failed configurations.

## 8. Interpreting and reporting results

Do not claim that all 75 configurations completed unless the result files contain all of them. The theoretical matrix is 75 configurations; a single successful classifier run is expected to produce up to 15 configurations.

The code prints dataset statistics and sampled label counts, but the exact source-file statistics depend on the CSVs supplied at runtime. For reproducible reporting, record:

- Exact filenames and paths (or a dataset manifest) used for the final run
- Combined row count after missing-label removal
- Number of feature columns before and after preprocessing
- Label names and class counts before and after sampling/rare-class removal
- Number of rows actually used for experiments
- KPCA reference rows and components
- Kernel and classifier parameters
- Test size and random state
- Successful and failed configurations
- The five metrics/timing fields for each successful configuration

Do not infer dataset row counts, feature counts, or class distributions from filenames or from an experiment workbook alone. Calculate or capture them from the exact final input files and run logs.

## 9. Important limitations

- The method uses class-wise sampling when more than 10,000 rows are available; results may not represent the full dataset distribution.
- KPCA is fitted on at most 1,200 training rows, not all training rows.
- The script uses two KPCA components, which is a substantial dimensionality reduction and may affect classifier performance.
- Non-numeric feature values are coerced to missing values, and columns that are entirely missing after conversion are removed.
- Classes with fewer than two sampled rows are dropped before stratified splitting; the console reports the number of rows dropped at this stage.
- The script suppresses `FutureWarning` messages. Other configuration failures are printed and skipped, and failed configurations are not included in the output CSV.
- Results depend on the exact CSV files, package versions, data cleaning, sampling, split, kernels, and model parameters.
- The output CSV contains only the current run's successful configurations unless results are preserved and combined manually.

## 10. Suggested project structure

```text
project/
├── data/
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   └── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── kpca_nids_spyder_one_algorithm.py
├── requirements.txt
├── README.md
└── results/
    ├── results_AdaBoost.csv
    ├── results_XGBoost.csv
    ├── results_CatBoost.csv
    ├── results_LightGBM.csv
    └── results_KNN.csv
```

This is a suggested organization. The `results/` files shown are examples of names to save manually; their presence is not implied.
