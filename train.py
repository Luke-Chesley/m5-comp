from model import build_tft,build_time_series_ds
from config import get_config

def train_model(config):
    train_dataloader, val_dataloader, _,_ = build_time_series_ds(config)

    trainer, tft,_ = build_tft(config)
    trainer.fit(
        tft,
        train_dataloader,
        val_dataloader,
        #ckpt_path="m5/checkpoints/1/epoch=2-val_loss=371.75-3.ckpt",
    )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
