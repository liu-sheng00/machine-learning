import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Setting up a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             recall_score, confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print("Start reading data...")
    data = pd.read_excel('Training set of anti-tumor drugs.xlsx')
    print("Data reading completed")

    features = data.iloc[:, 1:-1]
    labels = data.iloc[:, -1]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

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

    # Storing visualization data
    learning_curve_data = {}
    calibration_curve_data = {}
    roc_curve_data = {}

    # Train the model and evaluate performance
    for model_name, mp in model_params.items():
        print(f"Grid search in progress: {model_name}...")
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)

        # Output optimal parameters for grid search
        print(f"{model_name} Optimal parameters: {grid_search.best_params_}")

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])
        sensitivity = recall_score(y_test, y_pred)

        results[model_name] = {
            "best_params": grid_search.best_params_,
            "model": best_clf,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
            "sensitivity": sensitivity,
        }

        performance_data.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "AUC": auc,
            "Sensitivity": sensitivity,
        })

        # Save the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append({
            "Model": model_name,
            "True Negative": cm[0, 0],
            "False Positive": cm[0, 1],
            "False Negative": cm[1, 0],
            "True Positive": cm[1, 1]
        })

        # Save learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            best_clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
        )
        learning_curve_data[model_name] = {
            "train_sizes": train_sizes,
            "train_mean": np.mean(train_scores, axis=1),
            "test_mean": np.mean(test_scores, axis=1),
        }

        # Save calibration curve data
        y_proba = best_clf.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        calibration_curve_data[model_name] = {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }

        # Save ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_curve_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
        }

    # Visualization 1: Learning Curve
    plt.figure(figsize=(10, 6))
    for model_name, data in learning_curve_data.items():
        plt.plot(data["train_sizes"], data["train_mean"], label=f'{model_name} - Training')
        plt.plot(data["train_sizes"], data["test_mean"], label=f'{model_name} - Validation')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    # Save the image instead of displaying it
    plt.savefig('Learning Curve_All Models.png', dpi=300)
    print("The learning curve image has been saved as 'Learning Curve_All Models.png'")

    # Visualization 2: Calibration Curve
    plt.figure(figsize=(10, 6))
    for model_name, data in calibration_curve_data.items():
        plt.plot(data["prob_pred"], data["prob_true"], marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    # Save the image instead of displaying it
    plt.savefig('Calibration Curves_All Models.png', dpi=300)
    print("The calibration curve image has been saved as 'Calibration Curves_All Models.png'")

    # Visualization 3: ROC Curve
    plt.figure(figsize=(10, 6))
    for model_name, data in roc_curve_data.items():
        plt.plot(data["fpr"], data["tpr"], label=f'{model_name} (AUC = {results[model_name]["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    # Save the image instead of displaying it
    plt.savefig('ROC curve_all models.png', dpi=300)
    print("The ROC curve image has been saved as 'ROC curve_all models.png'")

    # Confusion Matrix Visualization
    plt.figure(figsize=(15, 12))  # Set the canvas size
    num_models = len(confusion_matrices)

    for idx, cm_data in enumerate(confusion_matrices, start=1):
        cm = confusion_matrix(y_test, results[cm_data["Model"]]["model"].predict(X_test))
        ax = plt.subplot(2, 2, idx)  # Set as a sub-image with 2 rows and 2 columns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    annot_kws={'size': 24},  # Modify the size of the confusion matrix numbers
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['True Negative', 'True Positive'], ax=ax)
        ax.set_title(f'{cm_data["Model"]}', fontsize=22)  # Modify the title，remove"Confusion Matrix"


    plt.tight_layout()  # Adjust the subplot layout so that titles and labels do not overlap
    plt.savefig('Confusion Matrix_All Models.png', dpi=300)  # Save the confusion matrix image
    print("The confusion matrix image has been saved as 'Confusion Matrix_All Models.png'")

    # Export performance indicators to Excel
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_excel("Model performance indicators.xlsx", index=False)

    # Export confusion matrix to Excel
    confusion_df = pd.DataFrame(confusion_matrices)
    confusion_df.to_excel("Confusion Matrix Results.xlsx", index=False)

    print("Model performance metrics have been saved to 'Model performance indicators.xlsx' in the file.")
    print("The confusion matrix results have been saved to 'Confusion Matrix Results.xlsx' in the file.")
