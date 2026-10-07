from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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

        y = y.astype(str).map({
            labels[0]: 0,
            labels[1]: 1
        })

    return X, y


def score(model, X, y):

    predictions = model.predict(X)

    return [
        round(accuracy_score(y, predictions), 4),
        round(
            precision_score(
                y,
                predictions,
                zero_division=0
            ),
            4
        ),
        round(
            recall_score(
                y,
                predictions,
                zero_division=0
            ),
            4
        ),
        round(
            f1_score(
                y,
                predictions,
                zero_division=0
            ),
            4
        )
    ]


def run():

    X, y = load()

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -------------------------------------------------
    # 1. Decision Tree vs Random Forest
    # -------------------------------------------------

    dt = DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )

    dt.fit(Xtr, ytr)

    print(
        "Decision Tree [accuracy, precision, recall, F1]:",
        score(dt, Xte, yte)
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(Xtr, ytr)

    print(
        "Random Forest [accuracy, precision, recall, F1]:",
        score(rf, Xte, yte)
    )

    print(
        "OOB score:",
        round(rf.oob_score_, 4),
        "| OOB error:",
        round(1 - rf.oob_score_, 4)
    )

    # -------------------------------------------------
    # 2. Effect of Number of Trees
    # -------------------------------------------------

    counts = [10, 25, 50, 100, 200]

    oob_errors = []
    accuracies = []

    for n in counts:

        model = RandomForestClassifier(
            n_estimators=n,
            max_features="sqrt",
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        oob_errors.append(
            1 - model.oob_score_
        )

        accuracies.append(
            accuracy_score(
                yte,
                model.predict(Xte)
            )
        )

    plt.figure(figsize=(9, 6))

    plt.plot(
        counts,
        oob_errors,
        marker="o",
        label="OOB Error"
    )

    plt.plot(
        counts,
        accuracies,
        marker="o",
        label="Test Accuracy"
    )

    plt.xlabel("Number of Trees")
    plt.ylabel("Value")

    plt.title(
        "Effect of Number of Trees - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "random_forest_number_of_trees.png",
        dpi=150
    )

    plt.close()

    # -------------------------------------------------
    # 3. Effect of Feature Subsampling
    # -------------------------------------------------

    options = ["sqrt", "log2", None]

    labels = [
        "sqrt",
        "log2",
        "all features"
    ]

    values = []

    for option in options:

        model = RandomForestClassifier(
            n_estimators=100,
            max_features=option,
            random_state=42,
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        values.append(
            accuracy_score(
                yte,
                model.predict(Xte)
            )
        )

    print(
        "Feature subsampling:",
        {
            label: round(value, 4)
            for label, value in zip(labels, values)
        }
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        values
    )

    plt.ylabel("Test Accuracy")

    plt.title(
        "Feature Subsampling - Placement Prediction"
    )

    plt.ylim(0, 1)

    for i, value in enumerate(values):

        plt.text(
            i,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()

    plt.savefig(
        OUT / "random_forest_feature_subsampling.png",
        dpi=150
    )

    plt.close()

    print("Lab 9 Completed Successfully")


if __name__ == "__main__":
    run()