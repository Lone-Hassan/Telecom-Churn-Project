
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from sklearn.preprocessing import StandardScaler



def evaluate_model(logistic_pipeline, X_test, y_test):
    def evaluate_model(logistic_pipeline, X_test, y_test):
        """
        Evaluates a trained logistic regression pipeline on test data using multiple classification metrics.

        Parameters:
            logistic_pipeline (Pipeline or model): A fitted scikit-learn pipeline or classifier with `predict` and `predict_proba` methods.
            X_test (pd.DataFrame or np.ndarray): Feature matrix for the test set.
            y_test (pd.Series or np.ndarray): True labels for the test set.

        Functionality:
            - Computes and prints Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
            - Displays a detailed classification report with class labels ('No Churn', 'Churn').
            - Plots a confusion matrix heatmap for visual inspection of prediction performance.

        Notes:
            - Assumes binary classification with positive class labeled as 1 ('Churn').
            - Uses `predict_proba` to compute ROC-AUC based on predicted probabilities.
            - Designed for business-facing churn prediction evaluation.

        Returns:
            None. Outputs are printed and visualized.
        """
    # Predict labels and probabilities
    y_pred = logistic_pipeline.predict(X_test)
    y_proba = logistic_pipeline.predict_proba(X_test)[:, 1]

    # Basic metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Classification report
    target_names = ['No Churn', 'Churn']
    print("\nClassification Report:\n", classification_report(y_test, y_pred,target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Logistic Regression')
    plt.tight_layout()
    plt.show()
    
def model_comparison(models, X_test, y_test):
    """
    Compares multiple trained classification models on a common test set using standard evaluation metrics.

    Parameters:
        models (dict): A dictionary of model name–model object pairs. Each model must implement `predict` and `predict_proba`.
        X_test (pd.DataFrame or np.ndarray): Feature matrix for the test set.
        y_test (pd.Series or np.ndarray): True labels for the test set.

    Returns:
        pd.DataFrame: A DataFrame containing Accuracy, Precision, Recall, F1 Score, and ROC-AUC for each model.

    Notes:
        - Assumes binary classification with positive class labeled as 1.
        - Uses predicted probabilities (`predict_proba`) to compute ROC-AUC.
        - Ideal for comparing tuned models like Logistic Regression, Random Forest, and Gradient Boosting.
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
    
    results_df = pd.DataFrame(results)
    
    return results_df
    


