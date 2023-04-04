import pandas as pd
import glob
import os
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

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

label_list = glob.glob("../data/0404_s200/**/*.txt", recursive=True)

for label_path in tqdm(label_list):
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

kfold = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kfold.split(df, groups=df['image_id'])):
    df.loc[val_idx, 'fold'] = fold
df['fold'] = df['fold'].astype('int')
df.to_csv("./data/data_0404_s200.csv", index=False)


df = pd.read_csv("./data/data_0404_s200.csv")
print(df['fold'].value_counts())
print(df[df['fold']==0]['image_id'].nunique())
print(df[df['fold']==1]['image_id'].nunique())
print(df[df['fold']==2]['image_id'].nunique())
print(df[df['fold']==3]['image_id'].nunique())
print(df[df['fold']==4]['image_id'].nunique())