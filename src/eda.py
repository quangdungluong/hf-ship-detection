import os
import glob
import os
import sys
sys.path.insert(0, os.getcwd())
from src.utils import read_image, get_image_shape
from tqdm import tqdm

folder = "../ship-detection/test"
w_list = []
h_list = []

for file in tqdm(glob.glob(f"{folder}/**/*.png", recursive=True)):
    image = read_image(file)
    h, w = get_image_shape(image)
    h_list.append(h)
    w_list.append(w)

print(f"Min w: {min(w_list)}, Max w: {max(w_list)}")
print(f"Min h: {min(h_list)}, Max h: {max(h_list)}")

folder = "../ship-detection/train"
w_list = []
h_list = []

for file in tqdm(glob.glob(f"{folder}/**/*.png", recursive=True)):
    image = read_image(file)
    h, w = get_image_shape(image)
    h_list.append(h)
    w_list.append(w)

print(f"Min w: {min(w_list)}, Max w: {max(w_list)}")
print(f"Min h: {min(h_list)}, Max h: {max(h_list)}")
