import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             recall_score, confusion_matrix, roc_curve, precision_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print("Start reading data...")
    data = pd.read_excel('Cardiotoxicity_Morgan_Fingerprint.xlsx')
    print("Data loading complete.")

    # Data preprocessing
    features = data.iloc[:, 1:-1]
    labels = data.iloc[:, -1]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # Define models and parameters
    model_params = {
        'RF': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100, 200]}
        },
        'GBoost': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]}
        },
        'LR': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': [0.1, 1, 10]}
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }
    }

    results = {}
    performance_data = []
    confusion_matrices = []

    # Store visualization data
    learning_curve_data = {}
    calibration_curve_data = {}
    roc_curve_data = {}

    # Train models and evaluate performance
    for model_name, mp in model_params.items():
        print(f"Performing grid search for: {model_name}...")
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)

        print(f"{model_name} best parameters: {grid_search.best_params_}")

        # Compute performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])
        sensitivity = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        precision = precision_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        results[model_name] = {
            "best_params": grid_search.best_params_,
            "model": best_clf,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "mcc": mcc,
        }

        performance_data.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "AUC": auc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "MCC": mcc,
        })

        print(f"{model_name} Metrics: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}, AUC={auc:.4f}, "
              f"Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}, Precision={precision:.4f}, MCC={mcc:.4f}")

        confusion_matrices.append({
            "Model": model_name,
            "True Positive": cm[1, 1],
            "True Negative": cm[0, 0],
            "False Positive": cm[0, 1],
            "False Negative": cm[1, 0],
        })

        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            best_clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
        )
        learning_curve_data[model_name] = {
            "train_sizes": train_sizes,
            "train_mean": np.mean(train_scores, axis=1),
            "test_mean": np.mean(test_scores, axis=1),
        }

        # Calibration curve
        y_proba = best_clf.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        calibration_curve_data[model_name] = {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_curve_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
        }

    # Visualization 1: Learning Curve
    plt.figure(figsize=(12, 8))
    model_colors = {
        'RF': '#1f77b4',
        'GBoost': '#ff7f0e',
        'LR': '#2ca02c',
        'SVM': '#d62728'
    }

    for model_name, data in learning_curve_data.items():
        color = model_colors[model_name]
        plt.plot(data["train_sizes"], data["train_mean"],
                 linewidth=3, linestyle='-', color=color,
                 label=f'{model_name} - Training')
        plt.plot(data["train_sizes"], data["test_mean"],
                 linewidth=3, linestyle='--', color=color,
                 label=f'{model_name} - Validation')

    plt.xlabel('Training Set Size', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.show()

    # Visualization 2: Calibration Curve
    plt.figure(figsize=(12, 8))
    for model_name, data in calibration_curve_data.items():
        plt.plot(data["prob_pred"], data["prob_true"],
                 marker='o', markersize=8, linewidth=3,
                 color=model_colors[model_name],
                 label=model_name)

    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2,
             color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.show()

    # Visualization 3: ROC Curve
    plt.figure(figsize=(12, 8))
    for model_name, data in roc_curve_data.items():
        plt.plot(data["fpr"], data["tpr"],
                 linewidth=3, color=model_colors[model_name],
                 label=f'{model_name} (AUC = {results[model_name]["auc"]:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--',
             linewidth=2, color='gray', label='Random')
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.show()

    # Visualization 4: Confusion Matrices
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({
        'font.size': 12,
        'font.weight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold'
    })

    for idx, cm_data in enumerate(confusion_matrices):
        ax = plt.subplot(2, 2, idx + 1)
        model_name = cm_data["Model"]
        tn = cm_data["True Negative"]
        fp = cm_data["False Positive"]
        fn = cm_data["False Negative"]
        tp = cm_data["True Positive"]

        matrix = np.array([[tn, fp], [fn, tp]])

        cax = ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.5)

        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, f'{val}', ha='center', va='center',
                    fontsize=16, color='black', weight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted 0', 'Predicted 1'], fontsize=12)
        ax.set_yticklabels(['Actual 0', 'Actual 1'], fontsize=12)
        ax.xaxis.set_ticks_position('bottom')

        ax.text(0.5, -0.3, model_name, transform=ax.transAxes,
                ha='center', va='center', fontsize=14, weight='bold')

    plt.tight_layout(pad=3.0)
    plt.show()

    # Output performance metrics to Excel
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_excel("Model_Performance_Cardiotoxicity.xlsx", index=False)

    print("Model performance metrics saved to 'Model_Performance_Cardiotoxicity.xlsx'.")
