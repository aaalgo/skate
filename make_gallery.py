#!/usr/bin/env python3
import pickle
import cv2
from gallery.gallery import Gallery

with open('data/meta.pkl', 'rb') as f:
    # we don't need leftover here
    groups, _ = pickle.load(f)

COLS = 5
MAX_SIZE = 400

gal = Gallery('gallery', cols=COLS+1)   # 1st column is individual name

for length, name, files in groups[:20]:
    files = list(files)[:COLS]    # keep at most 5 columns
                                  # files is a set
    # remember files is a list of (filename, path)
    gal.text(name)      # add a text column
    for fname, path in files:
        image = cv2.imread(path)
        H, W = image.shape[:2]
        L = max(H, W)
        if L > MAX_SIZE:    # down size bigger images
                            # for efficient loading of web page
            H = H * MAX_SIZE // L
            W = W * MAX_SIZE // L
            image = cv2.resize(image, (W, H))
        cv2.imwrite(gal.next(), image)  # gal.next() adds a image column
    for _ in range(COLS - len(files)):
        gal.text('')              # pad with empty columns
gal.flush()
