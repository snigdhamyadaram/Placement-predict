from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

TARGET = "placement_status"


def load():

    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    df = df[FEATURES + [TARGET]].dropna()

    # Convert college tier to numbers
    if not pd.api.types.is_numeric_dtype(df["college_tier"]):

        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(
        df[FEATURES],
        columns=["branch"],
        dtype=float
    )

    # Encode target
    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):

        labels = sorted(y.astype(str).unique())

        if len(labels) != 2:
            raise ValueError(
                "placement_status must be binary"
            )

        y = y.astype(str).map({
            labels[0]: 0,
            labels[1]: 1
        })

    return X, y


def shap_report(model, X, name):

    print(f"\nGenerating SHAP analysis for {name}...")

    # Use a smaller sample for faster SHAP calculation
    sample_size = min(1000, len(X))
    X_sample = X.sample(
        n=sample_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        values = (
            shap_values[1]
            if len(shap_values) > 1
            else shap_values[0]
        )

    elif len(getattr(shap_values, "shape", ())) == 3:

        values = shap_values[:, :, 1]

    else:

        values = shap_values

    # Feature importance table
    importance = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance.to_csv(
        OUT / f"{name}_shap_feature_importance.csv",
        index=False
    )

    # SHAP summary plot
    plt.figure()

    shap.summary_plot(
        values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}_shap_summary.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # Dependence plot for top feature
    top_feature = importance.iloc[0]["Feature"]

    plt.figure()

    shap.dependence_plot(
        top_feature,
        values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}_shap_dependence.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nTop features ({name}):")
    print(
        importance.head(10).to_string(
            index=False
        )
    )


def run():

    X, y = load()

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = {

        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    }

    rows = []

    for name, model in models.items():

        print("\n" + "=" * 40)
        print(f"Training {name.upper()}...")
        print("=" * 40)

        start_time = time.perf_counter()

        model.fit(Xtr, ytr)

        training_time = round(
            time.perf_counter() - start_time,
            4
        )

        predictions = model.predict(Xte)

        accuracy = round(
            accuracy_score(
                yte,
                predictions
            ),
            4
        )

        print(
            f"{name.upper()} | "
            f"Accuracy: {accuracy} | "
            f"Training Time: {training_time}s"
        )

        print("\nClassification Report:")
        print(
            classification_report(
                yte,
                predictions
            )
        )

        shap_report(
            model,
            Xte,
            name
        )

        rows.append([
            name,
            accuracy,
            training_time
        ])

    comparison = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Accuracy",
            "Training_Time_Seconds"
        ]
    )

    comparison.to_csv(
        OUT / "xgb_lightgbm_comparison.csv",
        index=False
    )

    print("\n" + "=" * 40)
    print("MODEL COMPARISON")
    print("=" * 40)

    print(comparison)

    print("\nLab 10 Completed Successfully")


if __name__ == "__main__":
    run()