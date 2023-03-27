import glob

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
import pickle

import numpy as np

folder = "../ship-detection/test"
w_list = []
h_list = []

for file in tqdm(glob.glob(f"{folder}/**/*.png", recursive=True)):
    image = Image.open(file)
    w, h = image.size
    h_list.append(h)
    w_list.append(w)

print(f"Min w: {min(w_list)}, Max w: {max(w_list)}")
print(f"Min h: {min(h_list)}, Max h: {max(h_list)}")

folder = "../ship-detection/train"
w_list = []
h_list = []

for file in tqdm(glob.glob(f"{folder}/**/*.png", recursive=True)):
    image = Image.open(file)
    w, h = image.size
    h_list.append(h)
    w_list.append(w)

print(f"Min w: {min(w_list)}, Max w: {max(w_list)}")
print(f"Min h: {min(h_list)}, Max h: {max(h_list)}")

anno = pickle.load(open('anno.pkl', 'rb'))
w_list, h_list = [], []
for key, value in anno.items():
    for box in value:
        w, h = box[2]-box[0], box[3]-box[1]
        w_list.append(w)
        h_list.append(h)
print(f"{min(w_list)} - {max(w_list)}")
print(f"{min(h_list)} - {max(h_list)}")
print(f"{sum(np.array(w_list) > 400)} - {len(w_list)}")
print(f"{sum(np.array(h_list) > 400)} - {len(h_list)}")