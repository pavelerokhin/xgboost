"""
This script performs a binary classification task using XGBoost algorithm. It reads a data file in XLSX or CSV format,
performs data cleaning and type casting, splits the data into training and testing sets, trains an XGBoost model, and
evaluates the model's performance using confusion matrices.

Parameters:
- file_path: str - File path of the data file (XLSX or CSV).
- target_variable_label: str - Label of the target variable in the data.
- missing_value_identifier: str - Identifier for missing values (not used in the script).
- categorical_features_labels: List[str] - Labels of categorical features in the data.
- numeric_features_labels: List[str] - Labels of numeric features in the data.
- features_to_drop: List[str] - Labels of features to be dropped from the data.

Functions:
- to_numeric(df: pd.DataFrame, label: str) -> None: Converts a column in the DataFrame
to numeric type, handling missing data.
- read_data_file(file_path: str) -> pd.DataFrame: Reads the data file based on its extension (
XLSX or CSV). Returns a pandas DataFrame. """

import pandas as pd
import xgboost as xgb
import re
import numpy as np
from sklearn.model_selection import GridSearchCV  # cross validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# parameters
df_file_path: str = "df2.xlsx"
target_variable_label: str = "Target"
missing_value_identifier: str = None
categorical_features_labels: [str] = ["Feature 1", "Feature 2"]
numeric_features_labels: [str] = ["Feature 3", "Feature 4"]
features_to_drop: [str] = ["Feature 5"]


def to_numeric(df: pd.DataFrame, label: str) -> None:
    try:
        df[label] = pd.to_numeric(df[label])
    except ValueError as e:
        exception_value = str(e)
        match = re.search(r'Unable to parse string "(.*)"', exception_value)
        if match:
            missing_data_label_found: str = match.group(1)
            df.loc[df[label] == missing_data_label_found, label] = 0
            to_numeric(df, label)


def read_data_file(file_path: str) -> pd.DataFrame:
    file_extension = file_path.split('.')[-1]

    if file_extension == 'xlsx':
        return pd.read_excel(file_path)
    elif file_extension == 'csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Only XLSX and CSV files are supported.")


# Input
df: pd.DataFrame = read_data_file(df_file_path)

# Data cleaning, type casting
# plug-in all the data cleaning manipulations here

df.drop(features_to_drop,
        axis=1,
        inplace=True)

for label in numeric_features_labels:
    to_numeric(df, label)

# Target variable, predictors, One-hot encoding
X: pd.DataFrame = df.drop(target_variable_label, axis=1).copy()  # Dropping the target variable
y: pd.DataFrame = df[target_variable_label].copy()  # Target variable
X_encoded: pd.DataFrame = pd.get_dummies(X, columns=categorical_features_labels)  # One-hot encoding

# Training/testing split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

# XGBoost init and fit (not optimal)
clf_xgb: xgb.XGBClassifier = xgb.XGBClassifier(objective='binary:logistic', missing=0, seed=42)

print("Fitting not optimized model...")
clf_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

# Predicting the target variable for the filled testing set
y_pred: np.ndarray = clf_xgb.predict(X_test)

# Generating the confusion matrix
cm: np.ndarray = confusion_matrix(y_test, y_pred)
print("Confusion metrics for not optimized model:")
print(cm)

# Optimization
# fill-in the senseful values for the parameters
param_grid: dict = {
    'max_depth': [3],
    'learning_rate': [0.1],
    'gamma': [0.25],
    'reg_lambda': [10.0],
    'scale_pos_weight': [1]
}

optimal_params_test: GridSearchCV = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42, subsample=0.9,
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0,
    n_jobs=10,
    cv=3
)

optimal_params_test.fit(X_train,
                        y_train,
                        early_stopping_rounds=10,
                        eval_metric='auc',
                        eval_set=[(X_test, y_test)],
                        verbose=False)

# Here iteratively repeat the optimisation in order to find the best parameters
best_params: dict = optimal_params_test.best_params_
print("Optimal parameters for the model:")
print(best_params)

gamma = best_params['gamma']
learning_rate = best_params['learning_rate']
max_depth = best_params['max_depth']
reg_lambda = best_params['reg_lambda']
scale_pos_weight = best_params['scale_pos_weight']

clf_xgb_optimal: xgb.XGBClassifier = xgb.XGBClassifier(seed=42, objective='binary:logistic',
                                                       gamma=gamma,
                                                       learn_rate=learning_rate,
                                                       max_depth=max_depth,
                                                       reg_lambda=reg_lambda,
                                                       scale_pos_weight=scale_pos_weight,
                                                       subsample=0.9,
                                                       colsample_bytree=0.5)

# Predicting the target variable for the filled testing set
y_pred2: np.ndarray = clf_xgb_optimal.predict(X_test)

# Generating the confusion matrix
cm2: np.ndarray = confusion_matrix(y_test, y_pred2)
print("Confusion metrics for the optimized model:")
print(cm2)
