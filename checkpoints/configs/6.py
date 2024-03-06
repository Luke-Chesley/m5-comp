from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss, MAE, MAPE, RMSE, SMAPE
from pytorch_forecasting.data import GroupNormalizer

import torch.nn as nn

LEVEL = 6


checkpoint_callback = ModelCheckpoint(
    dirpath="m5/checkpoints/" + str(LEVEL),
    filename="{epoch}-{val_loss:.2f}" + "-" + str(LEVEL),
    save_top_k=1,
    verbose=False,
    monitor="val_loss",
    mode="min",
    save_weights_only=False,  # Set to False to save the optimizer state as well # default
)
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")


def get_config():
    return {
        "level": LEVEL,
        "fc_dates": True,
        "target": "value",
        "max_pred_len": 28,
        "max_encoder_len": 30 * 250,
        "min_encoder_len": 30 * 2,
        "training_cutoff_quantile": 0.8,
        "min_pred_idx": 1941,
        "batch_size": 64,  # can change to accommodate for larger models
        "val_batch_size": 64,
        # tft params
        "tft_params": {
            "hidden_size": 64,  # can change
            "lstm_layers": 4,  # can change
            "dropout": 0.4,  # can change
            "output_size": 7,  # number of quantiles to map to
            "loss": QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            "attention_head_size": 8,  # can change
            "learning_rate": 0.002,  # can change
            "hidden_continuous_size": 32,  # can change
            "hidden_continuous_sizes": {},
            "log_interval": 10,
            "log_val_interval": 10,
            "log_gradient_flow": False,
            "optimizer": "Ranger",
            "reduce_on_plateau_patience": 4,
            "time_varying_categoricals_encoder": [],
            "time_varying_categoricals_decoder": [],
            "categorical_groups": {},
            "embedding_paddings": [],
            "monotone_constaints": {},
            "share_single_variable_networks": False,
            "causal_attention": True,
            "logging_metrics": nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]),
        },
        # trainer params
        "trainer_params": {
            "accelerator": "gpu",
            "max_epochs": 100,
            "enable_model_summary": True,
            "gradient_clip_val": 0.014,
            "limit_train_batches": 100,
            "callbacks": [checkpoint_callback, lr_logger, early_stop_callback],
            "logger": logger,
        },
    }
