import glob

import pandas as pd
from PIL import Image
from src.wbf import merge_wbf

Image.MAX_IMAGE_PIXELS = None
import pickle

from tqdm.auto import tqdm

df1 = pd.read_csv("./data/04041322.csv")
df2 = pd.read_csv("./data/04041304.csv")

labels = []
for df in [df1, df2]:
    labels.append(df.label.values)
images = sorted(glob.glob("../ship-detection/test/**/*.png", recursive=True))

with open("./data/image_test_size.pkl", 'rb') as fp:
    image_size = pickle.load(fp)

merged_results = []
for i, image_path in tqdm(enumerate(images)):
    w, h = image_size[i]
    list_predictions = [label[i] for label in labels]
    final_results = merge_wbf(list_predictions, 0.55, h, w, wbf_conf=0.01)
    merged_results.append(final_results)

df = pd.DataFrame({
    "id": df1.id.values,
    "label": merged_results
})

df.to_csv("./data/04041336.csv", index=False)