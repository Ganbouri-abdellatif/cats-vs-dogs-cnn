import os
import random
from shutil import copyfile

def create_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def split_data(main_dir, train_dir, valid_dir, test_dir=None, include_test_split=True, split_size=0.8):
    files = [f for f in os.listdir(main_dir) if os.path.getsize(os.path.join(main_dir, f)) > 0]
    random.shuffle(files)

    split = int(split_size * len(files))
    train_files = files[:split]

    if include_test_split:
        split_valid_test = int(split + (len(files) - split) / 2)
        valid_files = files[split:split_valid_test]
        test_files  = files[split_valid_test:]
    else:
        valid_files = files[split:]
        test_files = []

    for f in train_files:
        copyfile(os.path.join(main_dir, f), os.path.join(train_dir, f))
    for f in valid_files:
        copyfile(os.path.join(main_dir, f), os.path.join(valid_dir, f))
    for f in test_files:
        if include_test_split:
            copyfile(os.path.join(main_dir, f), os.path.join(test_dir, f))

    print(f"Split complete: {len(train_files)} train, {len(valid_files)} val, {len(test_files)} test.")
