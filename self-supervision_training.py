import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

from pytorch_lightning.loggers import WandbLogger

from data.dataset import SDOTilesDataset

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42  # So clever.
CHECKPOINT_DIR = "sim_siam"
LOG_EVERY_N_STEPS = 50
DATA_PATH = '/d0/euv/aia/preprocessed_ext/AIA_211_193_171/AIA_211_193_171_128x128'
EPOCHS = 16
DATA_STRIDE = 1
BATCH_SIZE = 4096
AUGMENTATION = 'single'
LOSS = 'contrast'  # 'contrast' or 'cos'
LEARNING_RATE = 0.1
PROJECTION_HEAD_SIZE = 128
PREDICTION_HEAD_SIZE = 128
EMBEDING_SIZE = 64


class SimSiam(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimSiamProjectionHead(512, 512, PROJECTION_HEAD_SIZE)
        self.prediction_head = SimSiamPredictionHead(PROJECTION_HEAD_SIZE, EMBEDING_SIZE, PREDICTION_HEAD_SIZE)
        self.criterion = NegativeCosineSimilarity()

        self.loss = LOSS
        self.loss_cos = NegativeCosineSimilarity()
        self.loss_contrast = NTXentLoss()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1, _) = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)

        loss_cos = 0.5 * (self.loss_cos(p0, z1) + self.loss_cos(p1, z0))
        loss_contrast = 0.5 * (self.loss_contrast(p0, z1) + self.loss_contrast(p1, z0))

        if self.loss == 'cos':
            loss = loss_cos
        else:
            loss = loss_contrast

        self.log('loss cos', loss_cos)
        self.log('loss contrast', loss_contrast)
        self.log('loss', loss)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


if __name__ == "__main__":

    pl.seed_everything(SEED, workers=True)

    dataset = SDOTilesDataset(data_path=DATA_PATH, augmentation=AUGMENTATION, data_stride=DATA_STRIDE)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=16,
    )

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project="hss_sss",
        name="sim_siam",
        log_model=True,
    )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="epoch-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_weights_only=False,
    )

    model = SimSiam()

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        deterministic=True,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=dataloader)
