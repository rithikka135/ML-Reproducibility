
import os
import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier


def main():
    # Read training parameters
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    train_params = params["train"]

    # Load training dataset
    train_data = pd.read_csv("data/processed/train.csv")

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # Create the Random Forest model
    model = RandomForestClassifier(
        n_estimators=train_params["n_estimators"],
        max_depth=train_params["max_depth"],
        random_state=train_params["random_state"]
    )

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")

    print("Model training completed successfully!")
    print("Number of trees:", train_params["n_estimators"])
    print("Maximum depth:", train_params["max_depth"])


if __name__ == "__main__":
    main()
