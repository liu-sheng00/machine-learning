import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    print("Start reading data...")
    # Reading training data
    train_data = pd.read_excel('Cardiotoxicity Morgan Fingerprint.xlsx')
    print("Training data reading completed")

    # Extracting features and labels
    features = train_data.iloc[:, 1:-1]  # The second to the second to last columns are features
    labels = train_data.iloc[:, -1]  # The last column is the label

    # Standardized features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Divide the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # Define the model and its parameters
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

    # Storing the trained model
    trained_models = {}

    # Training the model
    for model_name, mp in model_params.items():
        print(f"Training model: {model_name}...")
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Save the best model
        best_model = grid_search.best_estimator_
        trained_models[model_name] = best_model

        # Output the best parameters
        print(f"{model_name} best parameters: {grid_search.best_params_}")

        # Output the AUC on the test set
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"{model_name} Test Set AUC: {auc:.4f}")

    print("Model training completed！")

    # Reading prediction data
    print("Start reading forecast data...")
    predict_data = pd.read_excel('Pesticide Morgan Fingerprint.xlsx')
    print("Prediction data reading completed")

    # Extract compound names and Morgan fingerprints
    compounds = predict_data.iloc[:, 0]  # The first column is the compound name
    fingerprints = predict_data.iloc[:, 1:]  # The second to last column is the 1024-bit Morgan fingerprint

    # Standardized Morgan fingerprint
    fingerprints_scaled = scaler.transform(fingerprints)

    # Initialize the result list
    results = []

    # Make predictions for each model
    for model_name, model in trained_models.items():
        print(f"Using the model {model_name} Making predictions...")
        probabilities = model.predict_proba(fingerprints_scaled)[:, 1]  # Get the probability of the positive class
        results.append(probabilities)

    # Save the prediction results to DataFrame
    results_df = pd.DataFrame({
        "Compound": compounds,
        "RandomForest_Probability": results[0],  # Prediction probability of random forest
        "GradientBoosting_Probability": results[1],  # Prediction Probability with Gradient Boosting
        "LogisticRegression_Probability": results[2],  # Predicted Probability from Logistic Regression
        "SVM_Probability": results[3],  # Support Vector Machine Prediction Probability
    })

    # Save results to Excel file
    results_df.to_excel("Pesticide prediction probability.xlsx", index=False, engine='openpyxl')
    print("The prediction results have been saved to 'Pesticide prediction probability.xlsx'")