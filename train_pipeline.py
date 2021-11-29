import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks import ProgressBarBase
# from pytorch_lightning.metrics import Accuracy, Recall
from argparse import ArgumentParser
from model import Cat_Dog
import augmenters
import config
from tqdm import tqdm
from torchmetrics import Accuracy
from dataset import Cat_Dog_Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics import Metric
import cv2
import timm



class PrintCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print()
        print("Training epoch is started!")

    def on_train_epoch_end(self, trainer, pl_module):
        print()
        print("Training epoch is done.")


callbacks1 = PrintCallback()
exp_name = f'v3_effnet_{config.model_name}'
checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath=f'weights/{exp_name}/',  # this actually sq, not all
                                      filename='CatDog_{epoch:02d}_{train_loss:.4f}_{train_acc:.4f}'
                                               '{val_loss:.4f}_{val_acc:.4f}',
                                      save_top_k=20
                                      )

My_progressbar = ProgressBar(refresh_rate=1, process_position=0)
my_progressbarbase = ProgressBarBase()


class CatDogAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.correct += (preds == target).sum()
        self.total += len(target)

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


class CatDogDataModule(pl.LightningDataModule):
    def __init__(self, csv_file):
        super().__init__()
        self.train_augs = augmenters.train_augmentations()
        self.val_augs = augmenters.val_augmentations()
        self.data = pd.read_csv(csv_file).to_numpy()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_data, val_data = train_test_split(self.data, test_size=0.2, random_state=config.random_seed)
            self.train_dataset = Cat_Dog_Dataset(dataframe=train_data, augmentations=self.train_augs)
            self.val_dataset = Cat_Dog_Dataset(dataframe=val_data, augmentations=self.val_augs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.n_cpu,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.n_cpu,
                          pin_memory=True)


class CatDogPipeline(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        #self.model = model
        self.model = timm.create_model('efficientnet_b0', pretrained = True, num_classes = 2)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        # self.cls_acc = Balanced_acc()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        # print('labels: ', labels)

        predictions = self.model(images)
        # print('predictions :', predictions)

        train_loss = self.ce_loss(predictions, labels)
        train_acc = self.train_acc(torch.argmax(predictions, dim=1), labels.type(torch.int))

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_acc', train_acc, prog_bar=True)

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch

        predictions = self.model(images)

        val_loss = self.ce_loss(predictions, labels)
        val_acc = self.val_acc(torch.argmax(predictions, dim=1), labels.type(torch.int))

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

        return val_loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        lr = 0.001
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='min', factor=0.1,
                                                                    patience=5)

        return {
            'optimizer': self.opt,
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss'
        }


class PrintCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print()
        print("Training epoch is started!")

    def on_train_epoch_end(self, trainer, pl_module):
        print()
        print("Training epoch is done.")


callbacks1 = PrintCallback()
wandb_logger = WandbLogger(name=exp_name, project='camera_state')


class LitProgressBar(ProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description('running training ...')
        bar.leave = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        bar.leave = True
        return bar


bar = LitProgressBar()

lr_monitor = LearningRateMonitor(logging_interval='step')

model = Cat_Dog()

pipeline = CatDogPipeline(model)

trainer = pl.Trainer(gpus=config.n_gpus, accelerator='ddp', callbacks=[checkpoint_callback, callbacks1],
                     logger=[wandb_logger],
                     max_epochs=config.epochs, progress_bar_refresh_rate=40, num_sanity_val_steps=100)

loader = CatDogDataModule(config.csv_file)

trainer.fit(pipeline, loader)
