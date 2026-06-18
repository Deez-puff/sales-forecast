import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def prepare_data(featured_df, feature_columns, target_column='Total_Sales'):
    """
    Splits featured dataframe into X (features) and y (target).
    Uses last 12 months as test set, rest as training.
    """
    X = featured_df[feature_columns]
    y = featured_df[target_column]

    # ── Time-based split (don't shuffle time series data!) ──
    split_index = len(featured_df) - 12

    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model.
    Good for capturing overall trends.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Regressor.
    Good for capturing complex seasonal patterns.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    """
    Makes predictions using a trained model.
    """
    return model.predict(X)


def generate_future_forecast(model, featured_df, feature_columns, months_ahead=6):
    """
    Generates future sales forecasts for the next N months.
    Uses the last known data point to project forward.
    """
    last_row    = featured_df.iloc[-1].copy()
    last_sales  = featured_df['Total_Sales'].values
    forecasts   = []
    future_dates = []

    current_date = featured_df['Date'].iloc[-1]

    for i in range(months_ahead):
        # Move to next month
        next_month = current_date + pd.DateOffset(months=1)
        future_dates.append(next_month)

        # Build feature row for next month
        next_features = {
            'Year'           : next_month.year,
            'Month'          : next_month.month,
            'Quarter'        : next_month.quarter,
            'Season'         : get_season(next_month.month),
            'Month_Progress' : (next_month.month - 1) / 11,
            'Time_Index'     : last_row['Time_Index'] + i + 1,
            'Lag_1'          : last_sales[-1] if len(last_sales) > 0 else 0,
            'Lag_2'          : last_sales[-2] if len(last_sales) > 1 else 0,
            'Lag_3'          : last_sales[-3] if len(last_sales) > 2 else 0,
            'Lag_12'         : last_sales[-12] if len(last_sales) > 11 else 0,
            'Rolling_3'      : np.mean(last_sales[-3:]) if len(last_sales) > 2 else 0,
            'Rolling_6'      : np.mean(last_sales[-6:]) if len(last_sales) > 5 else 0,
            'Rolling_12'     : np.mean(last_sales[-12:]) if len(last_sales) > 11 else 0,
            'Is_Q4'          : 1 if next_month.quarter == 4 else 0,
            'Is_Year_End'    : 1 if next_month.month == 12 else 0
        }

        X_future = pd.DataFrame([next_features])[feature_columns]
        forecast  = model.predict(X_future)[0]
        forecasts.append(forecast)

        # Update for next iteration
        last_sales  = np.append(last_sales, forecast)
        current_date = next_month

    forecast_df = pd.DataFrame({
        'Date'            : future_dates,
        'Forecasted_Sales': forecasts
    })

    return forecast_df


def get_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)