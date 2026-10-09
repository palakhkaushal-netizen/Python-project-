# Machine Learning-Based Network Intrusion Detection

A Python machine learning project that investigates network traffic
classification using multiple classifiers and Kernel Principal Component
Analysis (Kernel PCA). The experiments compare different kernel
functions and train-test split ratios using standard classification
metrics.

> **Scope note:** This README is based on the project details discussed
> and the three CSV files provided. The Python implementation and
> experiment outputs were not included with the datasets, so this
> document describes the intended methodology rather than claiming that
> every experiment has already been completed.

## Datasets

The project includes these three network traffic CSV files:

  -----------------------------------------------------------------------------------------------------------------
  Dataset file                                                                    Columns                 File size
  ------------------------------------------------------------- ------------------------- -------------------------
  `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX(2).csv`                          79                  49.61 MB

  `Tuesday-WorkingHours.pcap_ISCX(2).csv`                                              79                 128.82 MB

  `Wednesday-workingHours.pcap_ISCX(2).csv`                                            79                 214.74 MB
  -----------------------------------------------------------------------------------------------------------------

The filenames are associated with the CIC-IDS2017 network traffic
dataset. Treat each file as a separate dataset input unless an
experiment explicitly combines them.

Before training, inspect each file's columns, data types, target/label
column, missing values, duplicate rows, infinite values, and class
distribution. Do not include the target column among the input features.

## Project Objective

The project studies how machine learning classifiers and Kernel PCA
kernel functions affect network traffic classification performance and
computational efficiency.

**Core research question:**

> How do different Kernel PCA kernel functions, machine learning
> algorithms, and train-test split ratios affect the classification
> performance and computational efficiency of network intrusion
> detection models?

## Machine Learning Algorithms

The five classifiers in the planned comparison are:

1.  XGBoost
2.  LightGBM
3.  CatBoost
4.  AdaBoost
5.  K-Nearest Neighbors (KNN)

## Kernel PCA

The planned Kernel PCA kernel functions are:

-   Linear
-   Polynomial
-   RBF (Radial Basis Function)
-   Sigmoid
-   Cosine

Kernel PCA can be computationally expensive on large datasets. Exact
kernel methods can require memory that grows quadratically with the
number of samples. Estimate memory requirements before running
experiments. If using sampling or an approximation such as Nyström,
document it clearly and distinguish those results from exact Kernel PCA
results.

## Experimental Design

  Parameter                      Configurations
  ------------------------------ ---------------------
  Classifiers                    5
  Kernel functions               5
  Test sizes                     0.2, 0.4, 0.6
  Equivalent train-test splits   80:20, 60:40, 40:60
  Planned configurations         75

The planned experiment matrix is:

**5 classifiers × 5 kernels × 3 test sizes = 75 configurations**

Run experiments sequentially by default to reduce memory pressure. Do
not interpret the planned configuration count as proof that all 75
experiments have been completed.

## Evaluation Metrics

Each experiment is intended to record:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Training time
-   Prediction time

Specify the averaging method for precision, recall, and F1-score (for
example, macro, weighted, or binary), and use a consistent method when
comparing compatible experiments. Record the dataset, target column,
split, kernel, model parameters, and preprocessing configuration with
every result.

## Development Environment

The project is intended to be developed and run in **Python using the
Spyder IDE**.

Libraries used or expected by this workflow may include:

-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `xgboost`
-   `lightgbm`
-   `catboost`

Keep a `requirements.txt` file containing the package versions used for
the final experiments.

## Recommended Workflow

1.  Load a dataset and inspect its structure.
2.  Identify the target column and class distribution.
3.  Clean and preprocess the data.
4.  Split into training and test sets.
5.  Fit preprocessing transformations on the training data only, then
    transform the test data to prevent data leakage.
6.  Apply the selected dimensionality-reduction method.
7.  Train the selected classifier.
8.  Evaluate predictions on the held-out test set.
9.  Record metrics, execution times, parameters, and any warnings or
    errors.
10. Repeat for the selected algorithms, kernels, and test sizes.
11. Export experiment results to CSV or Excel for analysis.

## Reproducibility

For each experiment, record:

-   Dataset filename and target column
-   Classifier and parameters
-   Kernel PCA kernel and parameters
-   Number of components
-   Train-test split and random seed
-   Preprocessing steps
-   Exact or approximate dimensionality-reduction method
-   Metric definitions and averaging strategy
-   Training and prediction time
-   Sampling, skipped steps, warnings, and errors

Use fixed random seeds where applicable. Report failed experiments as
failed, and never fabricate or manually alter metrics.

## Important Limitations

-   The CSV files do not establish the exact preprocessing, model
    parameters, or Kernel PCA settings used by the final code.
-   Kernel PCA on very large datasets may be impractical without a
    carefully documented sample limit or approximation.
-   Results depend on the dataset, target labels, preprocessing, split,
    and evaluation settings.
-   This README describes the project scope and intended experiments; it
    does not claim that the implementation or all experiments have been
    verified.

## Suggested Repository Structure

``` text
project/
├── data/
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   └── Thursday-WorkingHours-Morning-WebAttacks.csv
├── src/
│   ├── data_preprocessing.py
│   ├── kernel_pca.py
│   ├── models.py
│   ├── experiment_runner.py
│   └── evaluation.py
├── results/
├── requirements.txt
└── README.md
```

This is a suggested structure, not a listing of files confirmed to
exist.

## Current Repository Structure (organized)

```text
Python-project-/
├── README.md
├── code/        # root-level .py files
├── data/        # Kernel_PCA_ML_Experiment_Sheet_Final.xlsx
├── documents/   # Kernel PCA.txt
└── images/      # root-level .png experiment screenshots
```

## Running the Project

The exact command depends on the Python script in the repository. Open
the project in Spyder, configure the dataset paths and target column,
and run the experiment script. Once the implementation is finalized,
update this section with the exact script name and configuration
required to reproduce the experiments.


# Python-project-
Code-

import numpy as np
import pandas as pd

df1 = pd.read_csv("Tuesday.csv")
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv')
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')

dataset = pd.concat([df1, df2, df3], ignore_index=True)

dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=0,
    stratify=y
)

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)





from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nPrecision : {:.4f}".format(precision_score(y_test, y_pred, average='weighted')))
print("\nRecall : {:.4f}".format(recall_score(y_test, y_pred, average='weighted')))
print("\nF1 Score : {:.4f}".format(f1_score(y_test, y_pred, average='weighted')))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


Algorithms used -

9. XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

11. LightGBM
from lightgbm import LGBMClassifier
classifier = LGBMClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

13. CatBoost
from catboost import CatBoostClassifier
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

15. AdaBoost
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

17. k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)







classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
