from model import build_time_series_ds, build_tft
from config import get_config
from levels import level_id, level_id_dates
import numpy as np
import pandas as pd

from IPython.display import clear_output
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


def rmsse_scores(model_path):

    config_folder = model_path[-6]

    config = get_config()

    _, _, ds_train, _ = build_time_series_ds(config)

    dsp = ds_train.get_parameters()
    dsp["predict_mode"] = True

    df, _ = level_id_dates(config["level"], True, False)
    dft,_ = level_id_dates(config['level'],True,True)
    dft.time_idx += 1941
    dft.value = 0

    pred_outline = pd.concat([df, dft]).fillna(0)
    pred_outline.wday = pred_outline.wday.astype(str)
    pred_outline.month = pred_outline.month.astype(str)
    pred_outline.reset_index(drop=True,inplace=True)
    pred_outline.value = pred_outline.value.astype(float)

    new_prediction_data = TimeSeriesDataSet.from_parameters(dsp, pred_outline)

    tft = TemporalFusionTransformer.load_from_checkpoint(model_path)

    raw_preds = tft.predict(
        new_prediction_data, mode="raw", return_x=True,num_workers=32
    )  

    clear_output()

    labels = {
        outer_k: {v: k for k, v in tft.hparams.embedding_labels[outer_k].items()}
        for outer_k in tft.hparams.static_categoricals
    }
    groups = raw_preds[1]["groups"].cpu().numpy()

    labels_list = list(labels.values())
    new_groups = []

    for group in groups:
        new_group = []
        for i, item in enumerate(group):

            dict_to_use = labels_list[i % len(labels_list)]
            new_value = dict_to_use.get(item, None)
            new_group.append(new_value)
        new_groups.append(new_group)

    groups = np.array(new_groups, dtype=object)

    preds = raw_preds[0].prediction

    df_preds = pd.DataFrame()

    for g in range(len(groups)):
        p = preds[g][:, 3].cpu().numpy()
        col = tuple(groups[g])
        df_preds[col] = p

    df_preds = df_preds.round(0)
    df_preds.index = df_preds.index + 1942
    df_preds = df_preds[:-2].astype(int)

    ## true data -----------------------
    df, identifiers = level_id(config["level"], test=True)
    df_true = df.pivot_table(index="date", columns=identifiers, values="value")

    # loss -------------------------------------

    df_preds = df_preds.reset_index(drop=True)
    df_true = df_true.reset_index(drop=True)

    df_preds.columns = [
        col[0] if isinstance(col, tuple) and len(col) == 1 else col
        for col in df_preds.columns
    ]
    df_true.columns = [
        col[0] if isinstance(col, tuple) and len(col) == 1 else col
        for col in df_true.columns
    ]

    mse = ((df_true - df_preds) ** 2).mean()

    historical_diffs = df_true.diff().iloc[1:]
    scaling_factors = (historical_diffs**2).mean()

    scaling_factors = scaling_factors.replace(0, np.nan)

    rmsse = np.sqrt(mse / scaling_factors).mean()

    return rmsse, df_preds,df_true
