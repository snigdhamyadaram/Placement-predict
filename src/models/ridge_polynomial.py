import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# 1. Generate sample data
np.random.seed(42)

X = np.sort(6 * np.random.rand(100, 1) + 4)

y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])


# 2. Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 3. Create a degree-15 polynomial
poly_degree = 15

poly = PolynomialFeatures(degree=poly_degree)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 4. Standardize polynomial features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)


# 5. Define Ridge regularization values
lambdas = np.logspace(-4, 4, 200)

train_errors = []
test_errors = []


# 6. Train Ridge regression for every lambda
for lam in lambdas:

    ridge = Ridge(alpha=lam)

    ridge.fit(X_train_scaled, y_train)

    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)


# 7. Find lambda with minimum test error
best_index = np.argmin(test_errors)
best_lambda = lambdas[best_index]
best_test_error = test_errors[best_index]

print("Lab 6 Completed Successfully")
print("Polynomial Degree:", poly_degree)
print("Best Lambda:", best_lambda)
print("Minimum Test Error:", best_test_error)


# 8. Plot training and testing errors
plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    train_errors,
    label="Training Error",
    linewidth=2
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error",
    linewidth=2,
    linestyle="--"
)

plt.xscale("log")

plt.xlabel("Regularization Parameter (Lambda / Alpha)")
plt.ylabel("Mean Squared Error")

plt.title(
    "Ridge Regularization Path - Polynomial Regression (Degree 15)"
)

plt.legend()

plt.grid(True, which="both", linestyle="--")

plt.tight_layout()

plt.savefig(
    "reports/figures/ridge_regularization_error_curve.png",
    dpi=300
)

plt.show()