from levels import level_id_dates
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer 

import lightning.pytorch as pl


def build_training_data(config):
    df, identifiers = level_id_dates(config["level"], config["fc_dates"])

    target = config["target"]

    if identifiers:
        group_ids = identifiers
        static_categoricals = identifiers
    else:
        group_ids = []
        static_categoricals = []

    static_reals = []  # population for state etc

    time_varying_known_categoricals = ["wday", "month"]
    time_varying_unknown_categoricals = []

    variable_groups = {
        "event_days": [x for x in df.columns if x.startswith("event")],
        "snap_days": [x for x in df.columns if x.startswith("snap")],
    }

    df[[x for x in df.columns if x.startswith("event")]] = df[
        [x for x in df.columns if x.startswith("event")]
    ].astype("category")

    time_varying_known_reals = [
        x for x in df.columns if x.startswith("sin") or x.startswith("cos")
    ] + [
        "time_idx"
    ]  # probably price in here later

    time_varying_unknown_reals = [] + [
        config["target"]
    ]  # probably rolling mean in here later

    training_cutoff = round(df["time_idx"].quantile(config["training_cutoff_quantile"]))

    return (
        df,
        identifiers,
        target,
        group_ids,
        static_categoricals,
        static_reals,
        time_varying_known_categoricals,
        time_varying_unknown_categoricals,
        variable_groups,
        time_varying_known_reals,
        time_varying_unknown_reals,
        training_cutoff,
    )


def build_time_series_ds(config):
    (
        df,
        identifiers,
        target,
        group_ids,
        static_categoricals,
        static_reals,
        time_varying_known_categoricals,
        time_varying_unknown_categoricals,
        variable_groups,
        time_varying_known_reals,
        time_varying_unknown_reals,
        training_cutoff,
    ) = build_training_data(config)

    for l in [
        identifiers,
        static_categoricals,
        time_varying_unknown_categoricals,
    ]:
        df[l] = df[l].astype(str)

    df[identifiers] = df[identifiers].astype("category")

    df[variable_groups["event_days"]] = df[variable_groups["event_days"]].astype(
        "category"
    )
    df[variable_groups["snap_days"]] = df[variable_groups["snap_days"]].astype(
        "category"
    )

    df[target] = df[target].astype(float)
    df[["wday", "month"]] = df[["wday", "month"]].astype(str).astype("category")

    ds_train = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=group_ids,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        variable_groups=variable_groups,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        max_encoder_length=config["max_encoder_len"],
        min_encoder_length=config["min_encoder_len"],
        min_prediction_length=1,
        max_prediction_length=config["max_pred_len"],
        #min_prediction_idx=config['min_pred_idx'],
        target_normalizer=GroupNormalizer(
            groups=group_ids, transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    ds_val = TimeSeriesDataSet.from_dataset(
        ds_train, df, predict=True, stop_randomization=True
    )

    train_dataloader = ds_train.to_dataloader(
        train=True, batch_size=config["batch_size"], num_workers=31
    )
    val_dataloader = ds_val.to_dataloader(
        train=False, batch_size=config["val_batch_size"], num_workers=31
    )

    return train_dataloader, val_dataloader, ds_train, df # remove df


def build_tft(config):
    train_dataloader, val_dataloader, ds_train,_ = build_time_series_ds(config) # remove _

    trainer = pl.Trainer(**config["trainer_params"])

    tft = TemporalFusionTransformer.from_dataset(ds_train, **config["tft_params"])

    return trainer, tft,val_dataloader
