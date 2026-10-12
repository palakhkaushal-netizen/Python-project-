# -*- coding: utf-8 -*-
"""
KPCA + 5 Classifiers for CIC-IDS2017
Spyder-ready, memory-aware version.

IMPORTANT:
- Update DATA_DIR to the folder containing the three CSV files.
- Exact Kernel PCA on the full CIC-IDS2017 dataset is extremely memory-intensive.
- This script uses a stratified sample (MAX_ROWS) and fits each KPCA kernel on
  a smaller training reference sample (KPCA_FIT_ROWS).
- Results therefore describe this sampled/approximate experiment, not a
  full-data exact-KPCA experiment.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# Install once in Spyder's IPython console or the same environment:
# %pip install xgboost lightgbm catboost scikit-learn pandas numpy

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# =============================================================================
# 1. FILE PATHS
# =============================================================================
# Option A: Put this .py file in the same folder as the CSV files and leave "".
# Option B: Set DATA_DIR to the full folder path, e.g. r"D:\ML_Project\datasets".
DATA_DIR = ""

FILE_PATHS = [
    os.path.join(DATA_DIR, "Tuesday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Wednesday-workingHours.pcap_ISCX.csv"),
    os.path.join(DATA_DIR, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
]

# =============================================================================
# 2. MEMORY / EXPERIMENT SETTINGS
# =============================================================================
RANDOM_STATE = 0

# Start with 10,000 rows for a safer first run. Increase to 15,000 only if RAM allows.
MAX_ROWS = 10000

# KPCA fit matrix is approximately KPCA_FIT_ROWS x KPCA_FIT_ROWS.
# Start at 1000–1500; higher values use much more RAM.
KPCA_FIT_ROWS = 1200
N_COMPONENTS = 2

TEST_SIZES = [0.2, 0.4, 0.6]

# The five kernels requested for the project.
KERNELS = [
    {"name": "linear", "kernel": "linear", "params": {}},
    {"name": "polynomial", "kernel": "poly", "params": {"degree": 3}},
    {"name": "rbf", "kernel": "rbf", "params": {"gamma": 1.0}},
    {"name": "sigmoid", "kernel": "sigmoid", "params": {"gamma": 1.0}},
    {"name": "cosine", "kernel": "cosine", "params": {}},
]

# Run algorithms sequentially in this order.
# Choose ONLY ONE algorithm per run. Change this value manually:
# "XGBoost", "CatBoost", "LightGBM", "KNN", or "AdaBoost"
SELECTED_ALGORITHM = "AdaBoost"

OUTPUT_CSV = "kpca_selected_algorithm_results.csv"


def print_panel(test_size, kernel_name, algorithm_name, status="Running"):
    line = "-" * 46
    print("\n" + line)
    print("Performance Metrics")
    print(line)
    print("Test Size      :", test_size)
    print("Kernel         :", kernel_name)
    print("Algorithm      :", algorithm_name)
    print("Status         :", status)
    print(line)


def load_and_combine_datasets():
    print("Checking CSV files...")
    for path in FILE_PATHS:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "CSV file not found:\n"
                f"{path}\n\n"
                "Place the CSV files beside this Python script or update DATA_DIR."
            )

    frames = []
    for path in FILE_PATHS:
        print("Reading:", os.path.basename(path))
        # low_memory=False avoids mixed-dtype chunk inference warnings.
        frames.append(pd.read_csv(path, low_memory=False))

    print("Combining datasets...")
    data = pd.concat(frames, ignore_index=True)
    data.columns = data.columns.astype(str).str.strip()

    # CIC-IDS2017 label column is normally called "Label".
    if "Label" not in data.columns:
        raise ValueError(
            "Could not find a 'Label' column. Columns found: "
            + ", ".join(data.columns[-10:])
        )

    # Remove rows without a target label.
    data = data.dropna(subset=["Label"]).copy()
    data["Label"] = data["Label"].astype(str).str.strip()

    # Remove duplicate column names if any exist.
    data = data.loc[:, ~data.columns.duplicated()]

    print("Combined rows:", f"{len(data):,}")
    print("Feature columns:", data.shape[1] - 1)
    print("Labels:", data["Label"].nunique())

    return data


def stratified_sample(data):
    """Take an approximately proportional sample per class for manageable RAM."""
    if len(data) <= MAX_ROWS:
        return data.reset_index(drop=True)

    print(f"\nSampling approximately {MAX_ROWS:,} rows by class for memory safety...")
    fraction = MAX_ROWS / len(data)
    sampled = (
        data.groupby("Label", group_keys=False)
        .sample(frac=fraction, random_state=RANDOM_STATE)
    )

    # If rounding leaves a few rows short, add random rows from the remaining data.
    if len(sampled) < MAX_ROWS:
        remaining = data.drop(index=sampled.index, errors="ignore")
        needed = min(MAX_ROWS - len(sampled), len(remaining))
        if needed > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=needed, random_state=RANDOM_STATE)]
            )

    # If rounding slightly exceeds MAX_ROWS, randomly reduce it.
    if len(sampled) > MAX_ROWS:
        sampled = sampled.sample(n=MAX_ROWS, random_state=RANDOM_STATE)

    sampled = sampled.reset_index(drop=True)
    print("Rows used for experiments:", f"{len(sampled):,}")
    print("Sampled label counts:\n", sampled["Label"].value_counts().to_string())
    return sampled


def make_classifier(algorithm_name, n_classes):
    """Create only the selected classifier for this run."""
    if algorithm_name == "XGBoost":
        params = dict(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
            tree_method="hist", verbosity=0,
        )
        if n_classes > 2:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"
            params["num_class"] = n_classes
        else:
            params["eval_metric"] = "logloss"
        return XGBClassifier(**params)

    if algorithm_name == "CatBoost":
        return CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            verbose=False, thread_count=-1, random_seed=RANDOM_STATE,
            allow_writing_files=False,
        )
    if algorithm_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=100, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        )
    if algorithm_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if algorithm_name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)

    raise ValueError(
        f"Unknown SELECTED_ALGORITHM: {algorithm_name}. Choose XGBoost, "
        "CatBoost, LightGBM, KNN, or AdaBoost."
    )


def main():
    data = load_and_combine_datasets()
    data = stratified_sample(data)

    # Split features and target by the named Label column, not by last column position.
    X_df = data.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    y_text = data["Label"].astype(str)

    # Remove columns that are entirely missing/non-numeric.
    usable_columns = X_df.columns[X_df.notna().any(axis=0)]
    X_df = X_df[usable_columns]

    # Encode target labels.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    # Drop classes with fewer than 2 rows because stratified splitting requires
    # at least 2 examples per class. Normally this should affect few/no classes
    # in a reasonable sample, but the script reports any rows removed.
    counts = pd.Series(y).value_counts()
    valid_encoded_labels = counts[counts >= 2].index
    valid_mask = np.isin(y, valid_encoded_labels)
    dropped = int((~valid_mask).sum())
    if dropped:
        print(f"\nWarning: dropping {dropped} sampled rows from classes with fewer than 2 rows.")
        X_df = X_df.loc[valid_mask].reset_index(drop=True)
        y = y[valid_mask]
        # Re-encode after dropping rare classes so class IDs remain 0..n_classes-1.
        y = LabelEncoder().fit_transform(y)

    # Keep only numeric features; imputation/scaling are fit separately on train data
    # inside each split to reduce data leakage.
    results = []
    experiment_number = 0

    # Run only the algorithm selected at the top of this file.
    algorithm_name = SELECTED_ALGORITHM
    print("\n" + "=" * 70)
    print("STARTING SELECTED ALGORITHM:", algorithm_name)
    print("=" * 70)

    for test_size in TEST_SIZES:
        try:
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_df,
                y,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        except ValueError as exc:
            print(f"Skipping test size {test_size}: {exc}")
            continue

        # Fit imputer/scaler on training split only.
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
        X_test = scaler.transform(X_test).astype(np.float32, copy=False)

        # KPCA is fitted on a subset of training rows to avoid a huge
        # quadratic kernel matrix. Transform the whole split with that basis.
        fit_rows = min(KPCA_FIT_ROWS, len(X_train))
        rng = np.random.RandomState(RANDOM_STATE)
        fit_indices = rng.choice(len(X_train), size=fit_rows, replace=False)
        X_kpca_fit = X_train[fit_indices]

        for kernel_config in KERNELS:
            kernel_name = kernel_config["name"]
            print_panel(test_size, kernel_name, algorithm_name)

            try:
                kpca = KernelPCA(
                    n_components=N_COMPONENTS,
                    kernel=kernel_config["kernel"],
                    eigen_solver="arpack",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    **kernel_config["params"],
                )
                kpca.fit(X_kpca_fit)
                X_train_kpca = kpca.transform(X_train)
                X_test_kpca = kpca.transform(X_test)

                # Release large original feature arrays for this combination.
                # They are regenerated per split, not per kernel.
                classifier = make_classifier(algorithm_name, len(np.unique(y_train)))

                start_train = time.perf_counter()
                classifier.fit(X_train_kpca, y_train)
                training_seconds = time.perf_counter() - start_train

                start_predict = time.perf_counter()
                y_pred = classifier.predict(X_test_kpca)
                prediction_seconds = time.perf_counter() - start_predict

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )

                experiment_number += 1
                row = {
                    "Experiment": experiment_number,
                    "Algorithm": algorithm_name,
                    "Test Size": test_size,
                    "Kernel": kernel_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                    "Training Time (sec)": training_seconds,
                    "Prediction Time (sec)": prediction_seconds,
                    "Rows Used": len(data),
                    "KPCA Fit Rows": fit_rows,
                    "Components": N_COMPONENTS,
                }
                results.append(row)

                line = "-" * 46
                print(line)
                print("Performance Metrics")
                print(line)
                print(f"Test Size      : {test_size}")
                print(f"Kernel         : {kernel_name}")
                print(f"Accuracy       : {accuracy:.4f}")
                print(f"Precision      : {precision:.4f}")
                print(f"Recall         : {recall:.4f}")
                print(f"F1 Score       : {f1:.4f}")
                print(f"Training Time  : {training_seconds:.2f} sec")
                print(f"Prediction Time: {prediction_seconds:.2f} sec")
                print(line)
                print(f"{experiment_number}. {algorithm_name}: Done")
                print(line)

                # Save after every successful experiment so results survive a crash.
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

                del kpca, X_train_kpca, X_test_kpca, classifier, y_pred

            except Exception as exc:
                print(f"FAILED: {algorithm_name} | {kernel_name} | test={test_size}")
                print("Reason:", repr(exc))
                continue

        del X_train, X_test, X_train_df, X_test_df, y_train, y_test

    print("\nFINISHED ALGORITHM:", algorithm_name)
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    if not results:
        print("\nNo experiments completed. Check errors above and package installation.")
        return

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("ALL COMPLETED RESULTS")
    print("=" * 70)
    print(result_df.to_string(index=False))

    best_index = result_df["Accuracy"].idxmax()
    best = result_df.loc[best_index]
    print("\nCHAMPION CONFIGURATION")
    print("Algorithm :", best["Algorithm"])
    print("Kernel    :", best["Kernel"])
    print("Test Size :", best["Test Size"])
    print(f"Accuracy  : {best['Accuracy']:.6f}")
    print(f"F1-Score  : {best['F1-Score']:.6f}")
    print("\nSaved results to:", os.path.abspath(OUTPUT_CSV))
    print("Selected algorithm:", SELECTED_ALGORITHM)
    print("Expected configurations for this run: 1 algorithm x 5 kernels x 3 test sizes = 15.")
    print("To run another algorithm, change SELECTED_ALGORITHM at the top and run again.")
    print("Failed configurations, if any, are not included in the results CSV.")


if __name__ == "__main__":
    main()
