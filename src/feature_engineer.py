import pandas as pd
import numpy as np

def create_time_features(monthly_df):
    """
    Creates time-based features from the monthly sales dataframe.
    These features help the ML model learn trends and seasonality.
    """
    df = monthly_df.copy()

    # ── Basic Date Features ───────────────────────────────
    df['Year']    = df['Date'].dt.year
    df['Month']   = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter

    # ── Season Feature ────────────────────────────────────
    def get_season(month):
        if month in [12, 1, 2]:
            return 1   # Winter
        elif month in [3, 4, 5]:
            return 2   # Spring
        elif month in [6, 7, 8]:
            return 3   # Summer
        else:
            return 4   # Fall

    df['Season'] = df['Month'].apply(get_season)

    # ── Month Progress (0 to 1 scale) ─────────────────────
    df['Month_Progress'] = (df['Month'] - 1) / 11

    # ── Time Index (sequential number) ────────────────────
    df['Time_Index'] = range(len(df))

    # ── Lag Features (previous months sales) ──────────────
    df['Lag_1']  = df['Total_Sales'].shift(1)   # last month
    df['Lag_2']  = df['Total_Sales'].shift(2)   # 2 months ago
    df['Lag_3']  = df['Total_Sales'].shift(3)   # 3 months ago
    df['Lag_12'] = df['Total_Sales'].shift(12)  # same month last year

    # ── Rolling Average Features ──────────────────────────
    df['Rolling_3']  = df['Total_Sales'].shift(1).rolling(window=3).mean()
    df['Rolling_6']  = df['Total_Sales'].shift(1).rolling(window=6).mean()
    df['Rolling_12'] = df['Total_Sales'].shift(1).rolling(window=12).mean()

    # ── Is Q4 Flag (holiday season — usually highest sales) ──
    df['Is_Q4'] = (df['Quarter'] == 4).astype(int)

    # ── Is Year End Flag ──────────────────────────────────
    df['Is_Year_End'] = (df['Month'] == 12).astype(int)

    # ── Drop rows with NaN (from lag features) ────────────
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns():
    """
    Returns the list of feature columns used for training.
    """
    return [
        'Year', 'Month', 'Quarter', 'Season',
        'Month_Progress', 'Time_Index',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
        'Rolling_3', 'Rolling_6', 'Rolling_12',
        'Is_Q4', 'Is_Year_End'
    ]