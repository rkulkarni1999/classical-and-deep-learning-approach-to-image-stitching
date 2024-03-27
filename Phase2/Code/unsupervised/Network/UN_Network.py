import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate

import argparse
import pytorch_lightning as pl
from Dataset.dataset import SyntheticDataset, safe_collate


class HomographyModel(pl.LightningModule):
    def __init__(self, model_hparams):
        super(HomographyModel, self).__init__()
        self.modelparams = model_hparams    ## TO BE ADDED FOR TRANING PART
        # self.save_hyperparameters(model_hparams)
        self.model = Net()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = photometric_loss(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        # return {"loss": loss, "log": logs}
        return loss

    def validation_step(self, batch):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = photometric_loss(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    # def on_validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     logs = {"val_loss": avg_loss}
    #     return {"avg_val_loss": avg_loss, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.modelparams.learning_rate)

    def train_dataloader(self):
        train_set = SyntheticDataset(self.modelparams.train_path, rho=self.modelparams.rho)
        return DataLoader(
            train_set,
            num_workers=4,
            batch_size=self.modelparams.batch_size,
            shuffle=True,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):
        val_set = SyntheticDataset(self.modelparams.valid_path, rho=self.modelparams.rho)
        return DataLoader(
            val_set,
            num_workers=4,
            batch_size=self.modelparams.batch_size,
            collate_fn=safe_collate,
        )

def photometric_loss(delta, img_a, patch_b, corners):
    corners_hat = corners + delta

    # in order to apply transform and center crop,
    # subtract points by top-left corner (corners[N, 0])
    corners = corners - corners[:, 0].view(-1, 1, 2)

    h = get_perspective_transform(corners, corners_hat)
    h_inv = torch.inverse(h)
    patch_b_hat = warp_perspective(img_a, h_inv, (128, 128))

    return F.l1_loss(patch_b_hat, patch_b)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Block(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Net(nn.Module):
    def __init__(self, batch_norm=False):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            Block(2, 64, batch_norm),
            Block(64, 64, batch_norm),
            Block(64, 128, batch_norm),
            Block(128, 128, batch_norm, pool=False),
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4 * 2),
        )

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)  # combine two images in channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        delta = x.view(-1, 4, 2)
        return delta