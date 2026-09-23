import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# =========================================================
# PART A: BINARY CLASSIFICATION
# =========================================================

print("==========================================")
print("PART A: BINARY CLASSIFICATION")
print("==========================================")


# 1. Generate binary classification dataset
X_bin, y_bin = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_classes=2
)


# 2. Split data into 80% training and 20% testing
X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(
    X_bin,
    y_bin,
    test_size=0.20,
    random_state=42
)


# 3. Train Binary Logistic Regression
bin_model = LogisticRegression()

bin_model.fit(X_bin_train, y_bin_train)


# 4. Predict test data
y_bin_pred = bin_model.predict(X_bin_test)


# 5. Evaluate binary model
print("\n--- Binary Classification Report ---")

print(classification_report(y_bin_test, y_bin_pred))

print(
    "Binary Accuracy:",
    accuracy_score(y_bin_test, y_bin_pred)
)


# 6. Visualize binary decision boundary
plt.figure(figsize=(8, 6))

xx, yy = np.meshgrid(
    np.linspace(X_bin[:, 0].min() - 1, X_bin[:, 0].max() + 1, 200),
    np.linspace(X_bin[:, 1].min() - 1, X_bin[:, 1].max() + 1, 200)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = bin_model.predict(grid_points)

Z = Z.reshape(xx.shape)

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap=plt.cm.coolwarm
)

plt.scatter(
    X_bin_test[:, 0],
    X_bin_test[:, 1],
    c=y_bin_test,
    edgecolors="k",
    cmap=plt.cm.coolwarm
)

plt.title("Binary Logistic Regression Decision Boundary")

plt.xlabel("Feature 1 (e.g., CGPA)")
plt.ylabel("Feature 2 (e.g., Aptitude Score)")

plt.tight_layout()

plt.savefig(
    "reports/figures/binary_logistic_decision_boundary.png",
    dpi=300
)

plt.show()


# =========================================================
# PART B: MULTICLASS CLASSIFICATION
# =========================================================

print("\n==========================================")
print("PART B: MULTICLASS CLASSIFICATION")
print("==========================================")


# 1. Generate multiclass dataset
X_multi, y_multi = make_blobs(
    n_samples=1500,
    n_features=2,
    centers=3,
    random_state=42
)


# 2. Split data into 80% training and 20% testing
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_multi,
    y_multi,
    test_size=0.20,
    random_state=42
)


# 3. Train Multinomial Logistic Regression
multi_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)

multi_model.fit(X_m_train, y_m_train)


# 4. Train One-vs-Rest Logistic Regression
ovr_model = LogisticRegression(
    solver="lbfgs",
    multi_class="ovr",
    max_iter=1000
)

ovr_model.fit(X_m_train, y_m_train)


# 5. Predict using multinomial model
y_m_pred = multi_model.predict(X_m_test)


# 6. Evaluate multinomial model
print("\n--- Multiclass Logistic Regression Report ---")

print(classification_report(y_m_test, y_m_pred))

print(
    "Multiclass Accuracy:",
    accuracy_score(y_m_test, y_m_pred)
)


# 7. Confusion Matrix
cm = confusion_matrix(y_m_test, y_m_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False
)

plt.title("Confusion Matrix - Multiclass Classification")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "reports/figures/multiclass_logistic_confusion_matrix.png",
    dpi=300
)

plt.show()


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n==========================================")
print("Lab 7 Completed Successfully")
print("==========================================")