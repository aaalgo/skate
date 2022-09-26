#!/usr/bin/env python3
import os
import pickle
from collections import defaultdict
import subprocess as sp
import pandas as pd

# The goal of this program is:
#   - gather all images
#   - divide them into two parts
#   - part A: grouped by individuals, and contain individuals with >=2 images
#             these will be used as the positive set.
#   - part B: leftover images, whose individuals have only one image
#             these will be used as negative set.
#             We'll further divide the two parts into traing and testing set
#             but not here.

df = pd.read_csv('data/skatespotter.csv')

# sp.check_output runs a command and
#   - asserts that it is successful
#   - return the output as a bytes object
# when shell=True, the 1st argument is taken as a shell command
buffer = sp.check_output('find ./data/images/ -type f', shell=True)

paths = []
# we want to decode the buffer and add paths 
# use .decode to turn bytes into str
for path in buffer.decode('ascii').split('\n'):
    if len(path) > 0:
        paths.append(path)


# the CSV fileeee uses filename
# so we need to be able to map filenames to full paths
name_to_path = {}

for f in paths:
    fname = os.path.basename(f)         # this remove the directory portion
    assert not fname in name_to_path
    name_to_path[fname] = f

individuals = defaultdict(lambda:[])    # look for the usage of defaultdict
                                        # why don't I use {}?

# iterate through the dataframe
# gather paths belong to the same individual to lists
for _, row in df.iterrows():
    fname = row.Filename
    ind = row.Individual
    path = name_to_path.get(fname, None)
    # notice these checks
    # I use these checks to guarantee the output is coherent
    if path is None:
        continue
    if not isinstance(ind, str):
        continue
    if len(ind) == 0:
        continue
    individuals[ind].append((fname, path))

groups = []
for k, v in individuals.items():
    # v is a list of (fname, path)
    s = set(v)  # remove duplicates
    if len(s) <= 1:
        continue
    groups.append((len(s), k, s))   # add length so we can sort

# we could have inserted (k, s) to groups and
# used sort(key=lamda x: len(x[1])), without adding the length field.
# But key is to be invoked repeatedly. For a list of N items,
# key is expected to be invoked on each item for log(N) times.
# Doing len() log(N) timess is not efficient.  That's why we saved the length.
groups.sort(key=lambda x: x[0], reverse=True)

C = 0
for l, k, v in groups:
    for fname, path in v:
        # this is how one removes an entry from a dict
        # remove all paths belong to individuals with more than one images
        del name_to_path[fname]
        C += 1

leftover = list(name_to_path.values())

print("Individuals with dups:", len(groups), "images:", C)
print("Other images:", len(leftover))

with open('data/meta.pkl', 'wb') as f:
    pickle.dump((groups, leftover), f)


