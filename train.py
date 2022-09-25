#!/usr/bin/env python3
import sys
import os
from glob import glob
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Named tensors and all their associated*")
warnings.filterwarnings("ignore", ".*charset_normalizer.*")
warnings.filterwarnings("ignore", ".*is smaller than the logging interval.*")
import numpy as np
import logging
import random
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchmetrics import AUROC
import pytorch_lightning as pl
from config import *

def loss_with_logits (logits, labels):
    #return F.logsigmoid(logits)
    p1 = torch.clip(torch.sigmoid(logits), 0.00001, 0.99999)
    return torch.mean(- (1.0 - labels) * torch.log(1.0 - p1) - labels * torch.log(p1))

class Model (pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = resnet18(num_classes=512)
        self.auroc = AUROC(2)

    def forward (self, images):
        return self.net(images)

    def step_impl (self, batch, prefix):
        labels, images1, images2 = batch
        ft1 = self.forward(images1)
        ft2 = self.forward(images2)

        logits = torch.sum(ft1 * ft2, dim=-1)

        loss = loss_with_logits(logits, labels) #F.binary_cross_entropy_with_logits(logits, labels)

        self.log("%s_loss" % prefix, loss, batch_size=labels.shape[0], on_step=False, on_epoch=True)
        return loss, labels.detach(), logits.detach()

    def training_step (self, batch, batch_idx):
        loss, _, _ = self.step_impl(batch, 'train')
        return loss

    def validation_step (self, batch, batch_idx):
        loss, labels, logits =  self.step_impl(batch, 'val')
        self.val_labels.append(labels)
        self.val_scores.append(logits)
        return loss

    def on_validation_epoch_start (self):
        self.val_labels = []
        self.val_scores = []
        pass

    def on_train_epoch_end (self):
        labels = torch.cat(self.val_labels)
        scores = torch.cat(self.val_scores)
        auc = self.auroc(scores, labels)
        epoch = self.trainer.current_epoch
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get('train_loss', -1)
        val_loss = metrics.get('val_loss', -1)
        logging.info("%03d: T: %.3f V: %.3f %.3f" % (epoch,
            train_loss, val_loss, auc))

    def configure_optimizers (self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)

def image_batch (images):
    V = np.stack(images)    #   B H W C
    V = np.moveaxis(V, 3, 1)
    return torch.from_numpy(V).float()

class DataLoader:
    def __init__ (self, meta, batch, is_train):
        super().__init__()
        self.batch = batch
        self.is_train = is_train
        self.groups = meta[0]
        self.leftover = meta[1]
        self.samples = None
        random.seed(DEFAULT_DATALOADER_SEED)
        self.state = random.getstate()
        self.sample()
        self.offset = None
        self.count_size = (len(self.samples) + batch - 1) // batch

    def sample (self):
        random.setstate(self.state)
        samples = []
        for g in self.groups:
            samples.append((1, random.sample(g, 2)))
        leftover = random.sample(self.leftover, len(self.leftover))
        for i, left in enumerate(leftover):
            samples.append((0, [left, leftover[(i+1)%len(leftover)]]))
        self.state = random.getstate()
        self.samples = samples

    def __len__ (self):
        return self.count_size

    def __iter__ (self):
        if self.is_train:
            self.sample()
        self.offset = 0
        return self

    def __next__ (self):
        if self.offset >= len(self.samples):
            raise StopIteration()
        batch = self.samples[self.offset:(self.offset + self.batch)]
        self.offset += len(batch)
        labels = []
        images1 = []
        images2 = []
        for label, (left, right) in batch:
            labels.append(label)
            images1.append(left)
            images2.append(right)

        return torch.tensor(labels), image_batch(images1), image_batch(images2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default='data/split0')
    #parser.add_argument("--valfrac", type=float, default=0.1)
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument("-o", "--output", type=str, default='models')
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    args = parser.parse_args()

    LOG_FMT = "[%(levelno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    with open(args.split, 'rb') as f:
        train_db = DataLoader(pickle.load(f), args.batch, True)
        val_db = DataLoader(pickle.load(f), args.batch, False)

    ck = pl.callbacks.ModelCheckpoint(dirpath=args.output, save_top_k=1, monitor="val_loss")

    trainer = pl.Trainer(gpus=1, callbacks=[ck], enable_progress_bar = True, max_epochs=args.epochs)

    model = Model()

    trainer.fit(model, train_db, val_db)

