import pandas as pd
import xgboost as xgb
import re
from sklearn.model_selection import GridSearchCV  # cross validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# parameters
df_file_path = "df.xlsx"
target_variable_label = "Target"
missing_value_identifier = None
categorical_features_labels = ["Feature 1", "Feature 2"]
numeric_features_labels = ["Feature 3", "Feature 4"]
features_to_drop = ["Feature 5"]


def to_numeric(df, label):
    try:
        df[label] = pd.to_numeric(df[label])
    except ValueError as e:
        exception_value = str(e)
        match = re.search(r'Unable to parse string "(.*)"', exception_value)
        if match:
            missing_data_label_found = match.group(1)
            df.loc[df[label] == missing_data_label_found, label] = 0
            to_numeric(df, label)


# Input
df = pd.read_excel(df_file_path)  # Reading the data from an Excel file

# Data cleaning, type casting
# plug-in all the data cleaning manipulations here

df.drop(features_to_drop,
        axis=1,
        inplace=True)

for label in numeric_features_labels:
    to_numeric(df, label)

# Target variable, predictors, One-hot encoding
X = df.drop(target_variable_label, axis=1).copy()  # Dropping the target variable
y = df[target_variable_label].copy()  # Target variable
X_encoded = pd.get_dummies(X, columns=categorical_features_labels)  # One-hot encoding

# Training/testing split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

# XGBoost init and fit (not optimal)
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=0, seed=42)

print("Fitting not optimized model...")
clf_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

# Predicting the target variable for the filled testing set
y_pred = clf_xgb.predict(X_test)

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion metrics for not optimized model:")
print(cm)

# Optimization
# fill-in the senseful values for the parameters
param_grid = {
    'max_depth': [3],
    'learning_rate': [0.1],
    'gamma': [0.25],
    'reg_lambda': [10.0],
    'scale_pos_weight': [1]
}

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', seed=42, subsample=0.9,
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0,
    n_jobs=10,
    cv=3
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)

# Here iteratively repeat the optimisation in order to find the best parameters

print("Optimal parameters for the model:")
print(optimal_params.best_params_)

gamma = optimal_params.best_params_['gamma']
learning_rate = optimal_params.best_params_['learning_rate']
max_depth = optimal_params.best_params_['max_depth']
reg_lambda = optimal_params.best_params_['reg_lambda']
scale_pos_weight = optimal_params.best_params_['scale_pos_weight']

clf_xgb_optimal = xgb.XGBClassifier(seed=42, objective='binary:logistic', gamma=gamma,
                             learn_rate=learning_rate, max_depth=max_depth, reg_lambda=reg_lambda,
                             scale_pos_weight=scale_pos_weight, subsample=0.9, colsample_bytree=0.5)

# Predicting the target variable for the filled testing set
y_pred2 = clf_xgb_optimal.predict(X_test)

# Generating the confusion matrix
cm2 = confusion_matrix(y_test, y_pred2)
print("Confusion metrics for the optimized model:")
print(cm2)
