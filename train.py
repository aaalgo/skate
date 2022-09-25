#!/usr/bin/env python3
import sys
import os
from glob import glob
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Named tensors and all their associated*")
warnings.filterwarnings("ignore", ".*charset_normalizer.*")
warnings.filterwarnings("ignore", ".*is smaller than the logging interval.*")
import math
import pandas as pd
import numpy as np
import logging
import random
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import pytorch_lightning as pl
from config import *

class Model (pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.net = resnet18(num_classes=2)

    def forward (self, images):
        return self.net(images)

    def step_impl (self, batch, prefix):
        images1, images2, labels = batch
        ft1 = self.forward(images1)
        ft2 = self.forward(images2)

        logits = torch.inner(ft1, ft2)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.log("%s_loss" % prefix, loss, batch_size=bs, on_step=False, on_epoch=True)
        return loss

    def training_step (self, batch, batch_idx):
        return self.step_impl(batch, 'train')

    def validation_step (self, batch, batch_idx):
        return self.step_impl(batch, 'val')

    def on_train_epoch_end (self):
        epoch = self.trainer.current_epoch
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get('train_loss', -1)
        val_loss = metrics.get('val_loss', -1)
        logging.info("%03d: T: %.3f V: %.3f %.3f" % (epoch,
            train_loss, val_loss, 0))

    def configure_optimizers (self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)

class DataLoader:
    def __init__ (self, meta, batch, is_train):
        super().__init__()
        self.batch = batch
        self.is_train = is_train
        groups = []
        for _, _, group in meta[0]:
            groups.append([path for _, path in group])

        leftover = [path for _, path in meta[1]]

        self.groups = groups
        self.leftover = leftover
        self.samples = None
        random.seed(DEFAULT_DATALOADER_SEED)
        self.state = random.getstate()
        if not is_train:
            self.sample()
        self.offset = None
        self.count_size = (len(self.samples) + batch - 1) // batch

    def sample ():
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
        if self.offset >= self.count_size:
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
        return batch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default='data/lm_split_0.csv')
    parser.add_argument("--anno", type=str, default=DEFAULT_ANNOTATION_FILE)
    #parser.add_argument("--valfrac", type=float, default=0.1)
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("-p", "--params_weight", type=float, default=0.01)
    parser.add_argument("-l", "--landmark", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default='models')
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-r", "--radius", type=float, default=5)
    args = parser.parse_args()

    LOG_FMT = "[%(levelno)d] %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    annotation, K = load_annotation(args.anno)
    logging.info("Found %d samples, %d keypoints." % (len(annotation), K))

    df = pd.read_csv(args.split)

    samples = [[],[]]
    for _, row in df.iterrows():
        pid = row['patient']
        split = row['split']
        landmarks = annotation[pid]
        if args.landmark is None:
            lm = landmarks
        else:
            lm = landmarks[args.landmark:(args.landmark+1), :]
        sample = {
            'pid': pid,
            'path': 'data/bin/%d.bin' % pid,
            'landmarks': lm,
        }
        samples[split].append(sample)

    train, val = samples
    logging.info("train: %d, val: %d" % (len(train), len(val)))

    if args.landmark is None:
        model = Model(K, args.params_weight)
    else:
        model = Model(1, args.params_weight)

    train_db = DataLoader("train", train, args.batch, False, args.radius)
    print("Train samples: %d" % train_db.stream.size())
    val_db = DataLoader("val", val, args.batch, False, args.radius)
    print("Val samples: %d" % val_db.stream.size())

    ck = pl.callbacks.ModelCheckpoint(dirpath=args.output, save_top_k=1, monitor="val_loss")

    trainer = pl.Trainer(gpus=1, callbacks=[ck], enable_progress_bar = True, max_epochs=args.epochs)
    trainer.fit(model, train_db, val_db)

