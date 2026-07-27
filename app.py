import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Make random values reproducible
np.random.seed(42)


# Create daily dates for one year
dates = pd.date_range(
    start="2023-01-01",
    end="2023-12-31",
    freq="D"
)


# Generate synthetic temperature data
temperatures = (
    20
    + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    + np.random.normal(0, 2, len(dates))
)


# Create a DataFrame
temperature_data = pd.DataFrame({
    "date": dates,
    "temperature": temperatures
})

print(temperature_data.head())


def create_lagged_features(series, lag=5):
    """
    Use previous temperature values to predict
    the current temperature.
    """

    series = np.asarray(series)

    feature_set = []
    target_values = []

    for i in range(lag, len(series)):
        # Previous 'lag' values
        feature_set.append(series[i - lag:i])

        # Current value
        target_values.append(series[i])

    return np.array(feature_set), np.array(target_values)


def split_data(
    temperature_series,
    start,
    training_window,
    forecast_horizon
):
    """Split data into training and testing sections."""

    training_data = temperature_series[
        start:start + training_window
    ]

    testing_data = temperature_series[
        start + training_window:
        start + training_window + forecast_horizon
    ]

    return training_data, testing_data


def prepare_features(data, lag):
    """Create features and target values."""

    X, y = create_lagged_features(data, lag)

    return X, y


def train_model(X_train, y_train):
    """Train a Linear Regression model."""

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def make_predictions(model, X_test):
    """Make predictions using the trained model."""

    return model.predict(X_test)


def calculate_error(y_true, y_pred):
    """Calculate Mean Squared Error."""

    return mean_squared_error(y_true, y_pred)


def perform_walk_forward_validation(
    temperature_series,
    dates,
    training_window=90,
    forecast_horizon=30,
    lag=5
):
    """Perform walk-forward validation."""

    mse_scores = []
    all_predictions = []
    all_actual_values = []
    all_prediction_dates = []

    for start in range(
        0,
        len(temperature_series)
        - training_window
        - forecast_horizon
        + 1,
        forecast_horizon
    ):
        # Split training and testing data
        training_data, testing_data = split_data(
            temperature_series,
            start,
            training_window,
            forecast_horizon
        )

        # Create training features
        X_train, y_train = prepare_features(
            training_data,
            lag
        )

        # Create testing features
        X_test, y_test = prepare_features(
            testing_data,
            lag
        )

        # Train the model
        model = train_model(X_train, y_train)

        # Predict test temperatures
        test_predictions = make_predictions(
            model,
            X_test
        )

        # Calculate error
        mse = calculate_error(
            y_test,
            test_predictions
        )

        mse_scores.append(mse)

        # Save results
        all_predictions.extend(test_predictions)
        all_actual_values.extend(y_test)

        # Find dates corresponding to y_test
        test_start = start + training_window

        prediction_dates = dates[
            test_start + lag:
            test_start + forecast_horizon
        ]

        all_prediction_dates.extend(prediction_dates)

    return (
        np.array(mse_scores),
        np.array(all_predictions),
        np.array(all_actual_values),
        pd.DatetimeIndex(all_prediction_dates)
    )


def plot_results(
    prediction_dates,
    actual_values,
    predicted_values
):
    """Plot actual and predicted temperatures."""

    plt.figure(figsize=(12, 6))

    plt.plot(
        prediction_dates,
        actual_values,
        label="Actual Temperatures",
        alpha=0.7
    )

    plt.plot(
        prediction_dates,
        predicted_values,
        label="Predicted Temperatures",
        alpha=0.7
    )

    plt.title(
        "Walk-Forward Validation for Temperature Prediction"
    )

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


temperature_series = temperature_data[
    "temperature"
].to_numpy()

mse_scores, predictions, actual_values, prediction_dates = (
    perform_walk_forward_validation(
        temperature_series=temperature_series,
        dates=temperature_data["date"],
        training_window=90,
        forecast_horizon=30,
        lag=5
    )
)

print("MSE for each validation window:")
print(mse_scores)

print("\nAverage MSE:")
print(np.mean(mse_scores))

print("\nRoot Mean Squared Error:")
print(np.sqrt(np.mean(mse_scores)))

plot_results(
    prediction_dates,
    actual_values,
    predictions
)