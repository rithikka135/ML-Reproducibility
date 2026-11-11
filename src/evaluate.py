
import os
import json
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def main():
    # Load testing dataset
    test_data = pd.read_csv("data/processed/test.csv")

    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # Load trained model
    model = joblib.load("models/iris_model.pkl")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted"))
    }

    # Create reports folder
    os.makedirs("reports", exist_ok=True)

    # Save metrics as JSON
    with open("reports/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Setosa", "Versicolor", "Virginica"]
    )

    with open("reports/classification_report.txt", "w") as file:
        file.write(report)

    # Display results
    print("\nModel Evaluation Results")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
