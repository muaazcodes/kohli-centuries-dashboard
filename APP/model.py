"""
Virat Kohli Strike Rate Prediction
Ridge Regression model trained on centuries dataset.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


def load_data(path="Virat kohli.zip"):
    return pd.read_csv(path)


def prepare_features(virat_dataset):
    y = virat_dataset["Strike Rate"]

    X = virat_dataset.drop(
        ["Strike Rate", "Number", "Score", "Balls",
         "Against", "Venue", "Series", "Host Nation"],
        axis=1
    )

    if "Score Category" in X.columns:
        X = X.drop("Score Category", axis=1)

    X = pd.get_dummies(
        X,
        columns=["Format", "Win", "Captain", "Not Out", "MOTM", "Position"],
        drop_first=True
    )

    return X, y


def build_pipeline():
    numeric_cols = ["Team Total", "Wickets lost", "Year", "Inning"]

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)],
        remainder="passthrough"
    )

    model = Ridge(alpha=1.0)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def train_and_evaluate(X, y, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return pipeline


def show_coefficients(pipeline, X):
    model = pipeline.named_steps["model"]
    coefficients = pd.Series(model.coef_, index=X.columns)
    print("\nFeature coefficients:")
    print(coefficients.sort_values(ascending=False))


def save_model(pipeline, columns, path="virat_ridge_model.pkl"):
    joblib.dump({"pipeline": pipeline, "columns": list(columns)}, path)
    print(f"\nModel saved to {path}")


def main():
    virat_dataset = load_data()
    X, y = prepare_features(virat_dataset)
    pipeline = build_pipeline()
    pipeline = train_and_evaluate(X, y, pipeline)
    show_coefficients(pipeline, X)
    save_model(pipeline, X.columns)


if __name__ == "__main__":
    main()