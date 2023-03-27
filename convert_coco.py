import glob
import pickle
import os
import json
import random

eval_list = glob.glob("../data/raw_images/eval/**/*.png", recursive=True)
anno = pickle.load(open('anno.pkl', 'rb'))
eval_result = []
eval_anno = {"annotations": [],
             "categories": [{"supercategory": "a","id": 0, "name": "ship"}],
             "images": []}
id = 0
for image_path in eval_list:
    base_f = os.path.basename(image_path)
    image_id = int(base_f.split('.')[0])
    for bbox in anno[base_f]:
        eval_result.append({
            "image_id": image_id,
            "bbox": bbox,
            "score": 0.9,
            "category_id": 0
        })

        eval_anno["annotations"].append({
            "image_id": image_id,
            "bbox": bbox,
            "category_id": 0,
            "id": id,
            "iscrowd": 0,
            "area": 0
        })
        eval_anno["images"].append({
            "filename": image_path,
            "id": image_id
        })

        id += 1

with open("results.json", 'w') as fp:
    json.dump(eval_result, fp)
with open("anno.json", 'w') as fp:
    json.dump(eval_anno, fp)