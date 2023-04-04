import pandas as pd
from src.wbf import merge_wbf
import glob
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm.auto import tqdm

df1 = pd.read_csv("./data/submission_5fold.csv")
df2 = pd.read_csv("./data/submission_4fold.csv")
df3 = pd.read_csv("./data/submission_v7x_4fold.csv")

labels = []
for df in [df1, df2, df3]:
    labels.append(df.label.values)
images = sorted(glob.glob("../ship-detection/test/**/*.png", recursive=True))

merged_results = []
for i, image_path in tqdm(enumerate(images)):
    img = Image.open(image_path)
    w, h = img.size
    list_predictions = [label[i] for label in labels]
    final_results = merge_wbf(list_predictions, 0.55, h, w, wbf_conf=0.01)
    merged_results.append(final_results)

df = pd.DataFrame({
    "id": df1.id.values,
    "label": merged_results
})

df.to_csv("./data/post_ensemble_v7x.csv", index=False)