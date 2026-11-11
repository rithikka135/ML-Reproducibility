
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main():
    # Read parameters from params.yaml
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]

    # Load the original dataset
    data = pd.read_csv("data/iris.csv")

    # Separate input features and target
    X = data.drop("target", axis=1)
    y = data["target"]

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Create the processed data folder
    os.makedirs("data/processed", exist_ok=True)

    # Save training data
    train_data = X_train.copy()
    train_data["target"] = y_train
    train_data.to_csv("data/processed/train.csv", index=False)

    # Save testing data
    test_data = X_test.copy()
    test_data["target"] = y_test
    test_data.to_csv("data/processed/test.csv", index=False)

    print("Data preparation completed successfully!")
    print("Training samples:", len(train_data))
    print("Testing samples:", len(test_data))


if __name__ == "__main__":
    main()
