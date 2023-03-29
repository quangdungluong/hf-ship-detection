import pandas as pd
import glob
import os

def read_txt(file_path):
    data = []
    with open(file_path) as fp:
        lines = fp.readlines()
        for line in lines:
            data.append(line)
    return ", ".join(data)

image_paths = []
image_ids = []
labels = []
is_bgs = []

label_list = glob.glob("../data/ship-detection-v1/**/*.txt", recursive=True)

for label_path in label_list:
    image_name = os.path.basename(label_path)
    image_id = image_name.split('_')[0]
    label = read_txt(label_path)
    is_bg = (label == "")
    
    image_paths.append(image_name)
    image_ids.append(image_id)
    labels.append(label)
    is_bgs.append(is_bg)

df = pd.DataFrame({
    "image_name": image_paths,
    "image_id": image_ids,
    "label": labels,
    "is_bg": is_bgs
})

df.to_csv("data.csv", index=False)


