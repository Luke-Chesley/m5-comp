import numpy as np
def feature_creation_every_row(df):
    df["date"] = pd.to_datetime(df["date"])

    # hour
    # df["hour"] = df.date.dt.hour # change in params.py to match
    df["sin_hour"] = np.sin(2 * np.pi * df.date.dt.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df.date.dt.hour / 24)
    # dayofweek
    # df["dayofweek"] = df.date.dt.dayofweek
    df["sin_day_week"] = np.sin(2 * np.pi * df.date.dt.day_of_week / 7)
    df["cos_day_week"] = np.cos(2 * np.pi * df.date.dt.day_of_week / 7)

    # dayofyear

    # df["dayofyear"] = df.date.dt.dayofyear
    df["sin_day_of_year"] = np.sin(2 * np.pi * df.date.dt.day_of_year / 365)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df.date.dt.day_of_year / 365)

    # week of year
    # df["weekofyear"] = df.date.dt.isocalendar().week
    df["sin_week_year"] = np.sin(2 * np.pi * df.date.dt.isocalendar().week / 7)
    df["cos_week_year"] = np.cos(2 * np.pi * df.date.dt.isocalendar().week / 7)

    # month
    # df["month"] = df.date.dt.month
    df["sin_month"] = np.sin(2 * np.pi * df.date.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df.date.dt.month / 12)

    # quarter
    # df["quarter"] = df.date.dt.quarter
    df["sin_quarter"] = np.sin(2 * np.pi * df.date.dt.quarter / 4)
    df["cos_quarter"] = np.cos(2 * np.pi * df.date.dt.quarter / 4)

    # year
    df["year"] = df.date.dt.year

    return df


import pandas as pd
