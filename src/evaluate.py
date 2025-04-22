import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score
)
from sklearn.dummy import DummyRegressor

def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000):
    """
    Compute 95% confidence intervals for RMSE, MAE, and R² using bootstrapping.

    Parameters:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.
        n_bootstrap (int): Number of bootstrap resampling iterations.

    Returns:
        np.ndarray: A (2, 3) array containing the 2.5th and 97.5th percentiles 
                    for RMSE, MAE, and R².
    """
    metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        mse = mean_squared_error(y_true[indices], y_pred[indices])
        mae = mean_absolute_error(y_true[indices], y_pred[indices])
        r2 = r2_score(y_true[indices], y_pred[indices])
        metrics.append((np.sqrt(mse), mae, r2))
    return np.percentile(metrics, [2.5, 97.5], axis=0)

def evaluate_model_with_baseline(model, X_train, y_train, X_test, y_test):
    """
    Trains a given model, evaluates it on the test set using bootstrapped confidence 
    intervals, and compares its performance to a baseline DummyRegressor.

    Parameters:
        model: A scikit-learn compatible regression model (must implement fit/predict).
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target vector for training.
        X_test (array-like): Feature matrix for testing.
        y_test (array-like): Target vector for testing.

    Prints:
        95% confidence intervals for RMSE, MAE, and R² of both the trained model 
        and a mean-based DummyRegressor baseline.
    """
    # Train and evaluate the given model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_ci = bootstrap_metrics(y_test.values, y_pred)

    print("Final Model - 95% Confidence Intervals:")
    print(f"RMSE: {model_ci[0][0]:.2f} to {model_ci[1][0]:.2f}")
    print(f"MAE:  {model_ci[0][1]:.2f} to {model_ci[1][1]:.2f}")
    print(f"R²:   {model_ci[0][2]:.3f} to {model_ci[1][2]:.3f}")

    # Train and evaluate the baseline model
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    y_baseline_pred = baseline.predict(X_test)

    baseline_ci = bootstrap_metrics(y_test.values, y_baseline_pred)

    print("\n Baseline DummyRegressor - 95% Confidence Intervals:")
    print(f"RMSE: {baseline_ci[0][0]:.2f} to {baseline_ci[1][0]:.2f}")
    print(f"MAE:  {baseline_ci[0][1]:.2f} to {baseline_ci[1][1]:.2f}")
    print(f"R²:   {baseline_ci[0][2]:.3f} to {baseline_ci[1][2]:.3f}")
