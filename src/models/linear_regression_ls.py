import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the ingestion function from your data pipeline
from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():

    # 1. Load Data
    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv")

    df = load_and_validate_data(DATA_PATH)

    # Remove rows where required values are missing
    df_clean = df.dropna(
        subset=["cgpa", "communication_skill_score", "salary_package_lpa"]
    ).copy()

    # 2. Define Input and Output
    # L = 2 input features
    feature_cols = ["cgpa", "communication_skill_score"]

    # M = 1 output
    target_col = "salary_package_lpa"

    X_raw = df_clean[feature_cols].values
    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print(
        f"Loaded {N} data points with input dimension "
        f"L = {X_raw.shape[1]} and output dimension M = {y.shape[1]}"
    )

    # 3. Create Design Matrix
    # Add a column of 1s for the intercept/bias
    #
    # X_design =
    # [1  cgpa  communication_skills]
    #
    X_design = np.hstack([np.ones((N, 1)), X_raw])

    # 4. Standard Least Squares / Normal Equation
    #
    # w = (X^T X)^(-1) X^T y
    #
    XT_X = np.dot(X_design.T, X_design)

    try:
        XT_X_inv = np.linalg.inv(XT_X)
    except np.linalg.LinAlgError:
        print("Matrix is singular. Using pseudo-inverse instead.")
        XT_X_inv = np.linalg.pinv(XT_X)

    XT_y = np.dot(X_design.T, y)

    w_optimal = np.dot(XT_X_inv, XT_y)

    # Display weights
    print("\n--- Optimal Model Parameters ---")

    print(f"Intercept (w0): {w_optimal[0, 0]:.4f}")
    print(
        f"Coefficient for {feature_cols[0]} (w1): "
        f"{w_optimal[1, 0]:.4f}"
    )
    print(
        f"Coefficient for {feature_cols[1]} (w2): "
        f"{w_optimal[2, 0]:.4f}"
    )

    # 5. Calculate Predictions
    y_pred = np.dot(X_design, w_optimal)

    # 6. Calculate Error
    # E_w = 0.5 * sum((prediction - actual)^2)
    E_w = 0.5 * np.sum((y_pred - y) ** 2)

    print(f"Minimized Error (E_w): {E_w:.4f}")

    # 7. Create folder for visualization
    os.makedirs("reports/figures", exist_ok=True)

    # 8. Create 3D Plot
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")

    # Actual data points
    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        color="blue",
        alpha=0.6,
        label="Actual Data Points"
    )

    # Create meshgrid
    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        20
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        20
    )

    x1_mesh, x2_mesh = np.meshgrid(x1_surf, x2_surf)

    # Regression plane
    #
    # y = w0 + w1*x1 + w2*x2
    #
    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        color="red",
        alpha=0.3,
        edgecolor="none"
    )

    # Labels
    ax.set_xlabel("CGPA (Feature 1)")
    ax.set_ylabel("Communication Skills (Feature 2)")
    ax.set_zlabel("Salary Package LPA (Target)")

    ax.set_title(
        "Linear Regression via Standard Least Squares "
        "(P=1, L=2, M=1)"
    )

    plt.tight_layout()

    # 9. Save figure
    output_path = "reports/figures/linear_regression_3d_plane.png"

    plt.savefig(output_path)

    plt.close()

    print(
        f"\n-> Successfully saved 3D regression plot to "
        f"{output_path}"
    )


if __name__ == "__main__":
    train_linear_regression_ls()