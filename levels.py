import os
import pandas as pd
from feature_creation import feature_creation_every_row
from utils import reduce_mem_usage

# TODO: Add prices and dates


def level_id(level:int,test=False)->pd.DataFrame:
    path = "data/sales_train_evaluation.csv"

    if not os.path.exists(path):
        path = "m5/data/sales_train_evaluation.csv"

    df = pd.read_csv(path)

    if test:
        df = pd.read_csv("data/sales_test_evaluation.csv")

    df = df.rename(
        columns={
            col: col.replace("d_", "") if "d_" in col else col for col in df.columns
        }
    )


    start_date = "2016-05-23" if test else "2011-01-29"
 
    # Calculate the number of days (columns to rename minus the non-date columns)
    num_days = len(df.columns) - 5  # Adjust 5 based on your non-date columns

    # Generate a date range starting from 'start_date'
    date_range = pd.date_range(start=start_date, periods=num_days)

    # Create a mapping from old column names (numbers) to new column names (dates)
    # Preserving the first 5 column names (item_id, dept_id, cat_id, store_id, state_id) as they are not dates
    date_mapping = {
        str((i+1942) if test else (i+1)): date.strftime("%Y-%m-%d") for i, date in enumerate(date_range)
    }

    # Update the first 5 column names in the mapping to keep them unchanged
    for col in df.columns[:5]:
        date_mapping[col] = col

    # Rename the columns using the mapping
    df.columns = [
        date_mapping[col] if col in date_mapping else col for col in df.columns
]



    if level == 1:
        t_df = df.drop(
            ["item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1
        ).sum(axis=0)
        return (
            pd.DataFrame(t_df, columns=["value"])
            .reset_index()
            .rename(columns={"index": "date"})
        ),[]

    # 2-9
    gb_dict = {
        2: [["item_id", "dept_id", "cat_id", "store_id"], ["state_id"],["state_id"]],
        3: [["item_id", "dept_id", "cat_id", "state_id"], ["store_id"],["store_id"]],
        4: [["item_id", "dept_id", "store_id", "state_id"], ["cat_id"],["cat_id"]],
        5: [["item_id", "cat_id", "store_id", "state_id"], ["dept_id"],["dept_id"]],
        6: [["item_id", "dept_id", "store_id"], ["state_id", "cat_id"],["state_id","cat_id"]],
        7: [["item_id", "cat_id", "store_id"], ["state_id", "dept_id"],["state_id","dept_id"]],
        8: [["item_id", "dept_id", "state_id"], ["store_id", "cat_id"],["store_id","cat_id"]],
        9: [["item_id", "cat_id", "state_id"], ["store_id", "dept_id"],["store_id","dept_id"]],
    }

    if level in gb_dict:    
        drop = gb_dict[level][0]
        group = gb_dict[level][1]
        identifiers = gb_dict[level][2]

        df = df.drop(drop, axis=1).groupby(group).sum().T

        df = df.reset_index()

        df = df.rename(columns={"index": "date"})

        df_long = df.melt(id_vars=["date"], var_name=group, value_name="value")

        return df_long, identifiers

    # 10-11
    agg_dict = {
        "cat_id": "first",
        "dept_id": "first",
        "state_id": "first",
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg_dict.update(
        {
            col: "sum"
            for col in numeric_cols
            if col not in ["dept_id", "cat_id", "state_id"]
        }
    )

    sort_dict = {
        10: [
            ["item_id"],
            ["item_id", "state_id", "cat_id", "dept_id"],
            ["cat_id", "item_id", "dept_id"],
        ],
        11: [["item_id", "state_id"], ["item_id", "state_id", "cat_id", "dept_id"],["cat_id", "item_id", "dept_id","state_id"]],
        12: [
            ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        ],
    }

    gb = sort_dict[level][0]
    idv = sort_dict[level][1]
    identifiers = sort_dict[level][2]

    df_t = df.groupby(gb, as_index=False).agg(agg_dict)
    df_long = df_t.melt(id_vars=idv, var_name="date", value_name="value")

    if level == 10:
        df_long.drop("state_id", axis=1, inplace=True)

    return df_long, identifiers

def level_id_dates(level:int,fc_dates:bool=False,test=False)->pd.DataFrame:
    df, identifiers = level_id(level,test)

    path = "data/calendar.csv"
    if not os.path.exists(path):
        path = "m5/data/calendar.csv"

    df_cal = pd.read_csv(path)

    df = df.merge(df_cal, left_on="date", right_on="date", how="left")
    df.drop(columns=["weekday", "wm_yr_wk"], inplace=True)
    df.date = pd.to_datetime(df.date)
    df = pd.get_dummies(
        df, columns=["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    )

    if fc_dates:
        df = feature_creation_every_row(df)

    df['time_idx'] = df.date.factorize()[0]

    df, end_mem = reduce_mem_usage(df)

    print(f'df memory usage: {round(end_mem,2)} Mb')

    return df, identifiers
