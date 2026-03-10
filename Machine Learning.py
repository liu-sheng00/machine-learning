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
from sklearn.decomposition import PCA
import warnings

# Ignore joblib compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# ===== New: Output Directory =====
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':
    print("Starting to read data...")
    # Read rheumatoid arthritis dataset (replace with your actual file path if needed)
    data = pd.read_excel('rheumatoid_arthritis_dataset.xlsx')
    print("Data reading completed")

    # ================== Data Preprocessing ==================
    features = data.iloc[:, 1:-1]
    labels = data.iloc[:, -1]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42
    )

    # ================== Model Configuration ==================
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
    model_colors = {
        'RF': '#1f77b4',
        'GBoost': '#ff7f0e',
        'LR': '#2ca02c',
        'SVM': '#d62728'
    }

    # ================== Model Training and Evaluation ==================
    for model_name, mp in model_params.items():
        print(f"Performing grid search: {model_name}...")
        grid_search = GridSearchCV(
            mp['model'], mp['params'], cv=5,
            n_jobs=-1, scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)

        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)
        y_proba = best_clf.predict_proba(X_test)[:, 1]

        # Save key data
        cm = confusion_matrix(y_test, y_pred)
        results[model_name] = {
            "best_params": grid_search.best_params_,
            "model": best_clf,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "auc": roc_auc_score(y_test, y_proba),
            "sensitivity": recall_score(y_test, y_pred),
            "specificity": cm[0, 0] / (cm[0, 0] + cm[0, 1]),
            "precision": precision_score(y_test, y_pred),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "y_proba": y_proba,
            "y_test": y_test.values
        }

        performance_data.append({
            "Model": model_name,
            **{k: v for k, v in results[model_name].items()
               if k not in ['model', 'best_params', 'y_proba', 'y_test']}
        })

        confusion_matrices.append({
            "Model": model_name,
            "True Positive": cm[1, 1],
            "True Negative": cm[0, 0],
            "False Positive": cm[0, 1],
            "False Negative": cm[1, 0],
        })

    # ================== Visualization 1: Learning Curves ==================
    plt.figure(figsize=(10, 10))
    for model_name in model_params.keys():
        train_sizes, train_scores, test_scores = learning_curve(
            results[model_name]['model'], X_train, y_train, cv=5,
            scoring='accuracy', n_jobs=-1
        )
        plt.plot(train_sizes, np.mean(train_scores, axis=1),
                 linestyle='-', linewidth=5, color=model_colors[model_name],
                 label=f'{model_name} - Training')
        plt.plot(train_sizes, np.mean(test_scores, axis=1),
                 linestyle='--', linewidth=5, color=model_colors[model_name],
                 label=f'{model_name} - Validation')
    plt.xlabel('Training Set Size', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right', framealpha=1)
    plt.tick_params(labelsize=14)
    plt.gca().set_facecolor('#f8f9fa')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ================== Visualization 2: Calibration Curves ==================
    plt.figure(figsize=(8, 8))
    for model_name in model_params.keys():
        prob_true, prob_pred = calibration_curve(
            results[model_name]['y_test'],
            results[model_name]['y_proba'],
            n_bins=10
        )
        plt.plot(prob_pred, prob_true, marker='o', markersize=10, linewidth=5,
                 color=model_colors[model_name], label=model_name)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4, label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=16, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('#f8f9fa')
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ================== Visualization 3: ROC Curves ==================
    plt.figure(figsize=(8, 8))
    for model_name in model_params.keys():
        fpr, tpr, _ = roc_curve(
            results[model_name]['y_test'],
            results[model_name]['y_proba']
        )
        plt.plot(fpr, tpr, linewidth=5, color=model_colors[model_name],
                 label=f'{model_name} (AUC = {results[model_name]["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=4, label='Random')
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('#f8f9fa')
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ================== Visualization 4: Confusion Matrices ==================
    fig = plt.figure(figsize=(22, 5))
    for idx, model_name in enumerate(model_params.keys()):
        ax = plt.subplot(1, 4, idx + 1)
        cm_data = confusion_matrices[idx]
        matrix = np.array([
            [cm_data["True Negative"], cm_data["False Positive"]],
            [cm_data["False Negative"], cm_data["True Positive"]]
        ])
        im = ax.imshow(matrix, cmap="Blues", alpha=0.7)
        # ===== Add color bar only for SVM =====
        if model_name == 'SVM':
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]}",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold", color="black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['0', '1'], fontsize=12, fontweight='bold')
        ax.set_yticklabels(['0', '1'], fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
        ax.set_ylabel("Actual" if idx == 0 else "", fontsize=12, fontweight='bold')
        ax.set_title(model_name, fontsize=14, fontweight='bold', pad=10)
        ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ================== Visualization 5: Prediction Deviation Analysis ==================
    plt.figure(figsize=(22, 5))
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'font.weight': 'bold'
    })
    for idx, model_name in enumerate(model_params.keys()):
        ax = plt.subplot(1, 4, idx + 1)
        y_proba = results[model_name]['y_proba']
        y_test = results[model_name]['y_test'].astype(float)
        deviation = np.abs(y_proba - y_test)
        correct = (y_proba >= 0.5) == y_test

        sc1 = ax.scatter(
            y_proba[correct], deviation[correct],
            c=deviation[correct], cmap='Greens',
            vmin=0, vmax=1, alpha=0.7,
            s=60, edgecolor='k', linewidth=0.5,
            label='Correct'
        )
        sc2 = ax.scatter(
            y_proba[~correct], deviation[~correct],
            c=deviation[~correct], cmap='Reds',
            vmin=0, vmax=1, alpha=0.9,
            s=80, marker='X', linewidth=1.5,
            label='Error'
        )

        sorted_idx = np.argsort(y_proba)
        coeffs = np.polyfit(y_proba[sorted_idx], deviation[sorted_idx], 3)
        ax.plot(y_proba[sorted_idx], np.poly1d(coeffs)(y_proba[sorted_idx]),
                color='navy', linestyle='--', linewidth=3,
                label='Trend')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Predicted Probability', labelpad=10)
        if idx == 0:
            ax.set_ylabel('|Deviation|', labelpad=10)
        else:
            ax.set_ylabel('')

        ax.tick_params(axis='both', width=2, length=6,
                       labelsize=12, colors='black')
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.8)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.8)
        ax.set_title(model_name, pad=15, fontsize=16)
        if idx == 0:
            ax.legend(fontsize=10, loc='upper right',
                      frameon=True, framealpha=0.9,
                      edgecolor='black')

    cax = plt.axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(sc2, cax=cax)
    cbar.set_label('Deviation Level', fontsize=12, labelpad=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=12, width=2, length=6, colors='black')

    plt.subplots_adjust(left=0.05, right=0.9,
                        wspace=0.3, bottom=0.15)
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_deviation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ================== Visualization 6: PCA Dimensionality Reduction ==================
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(22, 5))
    for idx, model_name in enumerate(model_params.keys()):
        ax = plt.subplot(1, 4, idx + 1)
        y_pred = results[model_name]['model'].predict(X_test)
        correct = (y_pred == y_test)

        scatter = ax.scatter(X_test_pca[correct, 0], X_test_pca[correct, 1],
                             c=y_test[correct], cmap='coolwarm', alpha=0.7,
                             s=50, edgecolors='k', zorder=2)
        ax.scatter(X_test_pca[~correct, 0], X_test_pca[~correct, 1],
                   color='black', alpha=0.7, s=50, edgecolors='k', zorder=2)

        if idx == 3:
            handles, labels = scatter.legend_elements()
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Wrong',
                                      markerfacecolor='black', markersize=10))
            ax.legend(handles, ['Class 0', 'Class 1', 'Wrong'],
                      loc='upper right', fontsize=8)

        ax.set_title(model_name, fontsize=14, pad=12)
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2' if idx == 0 else '', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    pd.DataFrame(performance_data).to_excel("model_performance_metrics.xlsx", index=False)
    print(f"All figures are saved in: {OUTPUT_DIR}/")
    print("Model performance metrics are saved to 'model_performance_metrics.xlsx'")