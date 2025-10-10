import pandas as pd
from category_encoders import OneHotEncoder 
from sklearn.preprocessing import StandardScaler

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

def prepare_modeling_data(filepath):
    """
    Loads, cleans, and preprocesses a churn dataset for modeling, applying time-based splitting,
    one-hot encoding, and feature scaling.

    Parameters:
        filepath (str): Path to the CSV file containing the dataset. Assumes a 'date' column for temporal ordering.

    Returns:
        X_train (np.ndarray): Scaled and encoded feature matrix for the training set.
        X_test (np.ndarray): Scaled and encoded feature matrix for the test set.
        y_train (pd.Series): Target labels for the training set.
        y_test (pd.Series): Target labels for the test set.
        feature_names (np.ndarray): Array of encoded feature names after one-hot encoding.

    Processing Steps:
        - Reads the CSV file and parses the 'date' column.
        - Sorts the data chronologically to preserve temporal integrity.
        - Drops metadata and target columns to isolate features.
        - Fills missing values with zero (basic imputation).
        - Splits the data into training and test sets using the 80th percentile of the date column.
        - Applies one-hot encoding to categorical features using `category_encoders.OneHotEncoder`.
        - Applies standard scaling to normalize feature distributions.

    Notes:
        - Assumes binary churn label is stored in 'churn_binary'.
        - Designed for time-series aware modeling to prevent data leakage.
        - Returns encoded feature names for traceability and interpretation.
    """
    
    df_model = pd.read_csv(filepath,parse_dates=['date'])
    df_model = df_model.sort_values('date')
    
    # Cleaning and transformation steps
    feature_cols = [col for col in df_model.columns if col not in [
    'year', 'month', 'circle', 'service_provider', 'value', 'unit', 'notes',
    'date', 'churn_binary', 'churn_severity'
    ]]
    X = df_model[feature_cols].fillna(0) # Simple missing value handling
    y = df_model['churn_binary']
    
    # Time-based split
    split_date = df_model['date'].quantile(0.8)
    train_mask = df_model['date'] < split_date
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    # One-hot encoding and scaling
    ohe = OneHotEncoder(use_cat_names=True)
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    
    return X_train, X_test, y_train, y_test, ohe.get_feature_names_out()
    

def pre_processing(filepath):
    """
    Loads and preprocesses a churn dataset from the specified CSV file, applying time-based splitting and basic feature cleaning.

    Parameters:
        filepath (str): Path to the CSV file containing the dataset. Assumes a 'date' column for temporal ordering.

    Returns:
        X_train (pd.DataFrame): Feature matrix for the training set.
        X_test (pd.DataFrame): Feature matrix for the test set.
        y_train (pd.Series): Target labels for the training set.
        y_test (pd.Series): Target labels for the test set.

    Processing Steps:
        - Reads the CSV file and parses the 'date' column.
        - Sorts the data chronologically by date.
        - Drops non-feature columns including metadata and target labels.
        - Fills missing values with zero (basic imputation).
        - Splits the data into training and test sets using the 80th percentile of the date column.

    Notes:
        - Assumes binary churn label is stored in 'churn_binary'.
        - Designed for time-series aware modeling to prevent data leakage.
    """

   
    df_model = pd.read_csv(filepath,parse_dates=['date'])
    df_model = df_model.sort_values('date')
    
    # Example cleaning and transformation steps
    feature_cols = [col for col in df_model.columns if col not in [
    'year', 'month', 'circle', 'service_provider', 'value', 'unit', 'notes',
    'date', 'churn_binary', 'churn_severity'
    ]]
    X = df_model[feature_cols].fillna(0) # Simple missing value handling
    y = df_model['churn_binary']
    # Time-based split
    split_date = df_model['date'].quantile(0.8)
    train_mask = df_model['date'] < split_date
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    
    return X_train,X_test,y_train,y_test

def optuna_tune_model(X, y, model_type, n_trials=50, n_splits=4):
    """
    Performs hyperparameter tuning using Optuna for a specified classification model with time-series cross-validation.

    Parameters:
        X (pd.DataFrame): Feature matrix sorted chronologically for time-series modeling.
        y (pd.Series): Target labels corresponding to X.
        model_type (str): One of 'logistic', 'random_forest', or 'gradient_boosting'.
        n_trials (int): Number of Optuna trials to run (default is 50).
        n_splits (int): Number of TimeSeriesSplit folds for cross-validation (default is 4).

    Returns:
        optuna.Study: The Optuna study object containing best parameters and optimization history.

    Functionality:
        - Defines model-specific hyperparameter search spaces.
        - Applies TimeSeriesSplit to preserve temporal integrity during evaluation.
        - Encodes categorical features using OneHotEncoder (`category_encoders`).
        - Scales features using StandardScaler.
        - Trains and evaluates the model on each fold using ROC-AUC.
        - Logs fold-wise scores and mean AUC for each trial to '../docs/model_decision_log.md'.

    Notes:
        - Logistic Regression uses 'liblinear' for L1 penalty and 'lbfgs' for L2.
        - Designed for binary classification with `predict_proba` support.
        - Logging is enabled for traceability and model selection transparency.
    """
    

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            C = trial.suggest_float("C", 0.01, 10.0, log=True)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            solver = "liblinear" if penalty == "l1" else "lbfgs"

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            max_depth = trial.suggest_int("max_depth", 3, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            max_depth = trial.suggest_int("max_depth", 3, 10)

        else:
            raise ValueError("Unsupported model_type")

        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]

            ohe = OneHotEncoder(use_cat_names=True)
            ss = StandardScaler()

            X_tr = ohe.fit_transform(X_tr)
            X_val = ohe.transform(X_val)

            X_tr = ss.fit_transform(X_tr)
            X_val = ss.transform(X_val)

            if model_type == "logistic":
                model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=2000)
            elif model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            elif model_type == "gradient_boosting":
                model = GradientBoostingClassifier(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth
                )

            model.fit(X_tr, y_tr)
            y_pred = model.predict_proba(X_val)[:, 1]
            
            scores.append(roc_auc_score(y_val, y_pred))
        
        mean_score = sum(scores) / len(scores)
        log_entry = f"Trial {trial.number} | Model: {model_type} | Fold AUCs: {scores} | Mean AUC: {mean_score:.4f}\n"

        with open('../docs/model_decision_log.md', "a") as f:
            f.write(log_entry)

        #print(log_entry.strip())


        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study
    