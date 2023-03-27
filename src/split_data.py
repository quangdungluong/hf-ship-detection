import os
import shutil
import glob
from sklearn.model_selection import train_test_split

image_list = glob.glob("../ship-detection/train/**/*.png", recursive=True)
train_list, val_list = train_test_split(image_list, test_size=0.2, random_state=42)
for file in train_list:
    shutil.copy(file, "../data/raw_images/train")
for file in val_list:
    shutil.copy(file, "../data/raw_images/eval")