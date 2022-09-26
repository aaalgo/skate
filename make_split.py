#!/usr/bin/env python3
import sys
import time
import random
import numpy as np
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2
from config import *

def pick (lst, idx):
    return [lst[i] for i in idx]

CACHE = {}

def load_image (path):
    global CACHE
    image = CACHE.get(path, None)
    if not image is None:
        return image
    try:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (DEFAULT_SIZE, DEFAULT_SIZE))
        CACHE[path] = image
        return image
    except:
        return None

def load_list (lst):
    ret = []
    for path in lst:
        image = load_image(path)
        if not image is None:
            ret.append(image)
    return ret

def load (groups, leftover):
    data_groups = []
    data_leftover = load_list(leftover)
    for _, _, group in tqdm(groups):
        grp = load_list([path for _, path in group])
        if len(grp) == 1:
            data_leftover.extend(grp)
        else:
            data_groups.append(grp)
    return data_groups, data_leftover

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="")
    args = parser.parse_args()

    with open('data/meta.pkl', 'rb') as f:
        groups, leftover = pickle.load(f)

    random.seed(args.seed)
    random.shuffle(groups)
    random.shuffle(leftover)

    kf1 = KFold(n_splits=5)
    kf2 = KFold(n_splits=5)

    for i, ((train_group, test_group), (train_left, test_left)) \
            in enumerate(zip(kf1.split(groups), kf2.split(leftover))):
        print("SPLIT %d" % i)
        with open('data/split%d' % i, 'wb') as f:
            pickle.dump(load(pick(groups, train_group),
                         pick(leftover, train_left)), f)
            pickle.dump(load(pick(groups, test_group),
                         pick(leftover, test_left)), f)

