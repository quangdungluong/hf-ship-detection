import pickle
import glob
import os
import pandas as pd
anno = pickle.load(open('anno.pkl', 'rb'))
image_list = glob.glob("../data/raw_images/eval/**/*.png", recursive=True)

df = pd.read_csv("eval.csv")
data_list = df.to_dict('records')
pred = {}
for data in data_list:
    if data['id'] not in pred:
        pred[data['id']] = []
    for p in data['label'].split(", "):
        conf, x1, y1, x2, y2 = map(float, p.split())
        pred[data['id']].append(['ship', conf, x1, y1, x2, y2])

def save_txt(data, save_path):
    with open(save_path, 'w') as fp:
        for line in data:
            fp.write(" ".join(map(str, line)) + "\n")

## Annotation + Prediction
for image_path in image_list:
    base_f = os.path.basename(image_path)
    anno_data = anno[base_f]
    for d in anno_data: d.insert(0, "ship")
    save_txt(anno_data, os.path.join("./results/anno", f"{base_f[:-4]}.txt"))

    pred_data = pred[base_f]
    save_txt(pred_data, os.path.join("./results/pred", f"{base_f[:-4]}.txt"))