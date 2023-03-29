import json
import glob
import pandas as pd

def run():
    df = pd.read_csv("./data/eval_v2.csv")
    ids = df['id'].values
    labels = df['label'].values

    predictions = []
    for id, label in zip(ids, labels):
        for l in label.split(', '):
            score, x0, y0, x1, y1 = map(float, l.split())
            predictions.append({
                'image_id': int(id.split('.')[0]),
                'category_id': 1,
                'bbox': [x0, y0, x1-x0, y1-y0],
                'score': score
            })

    with open("./data/result_val.json", 'w', encoding='utf-8') as fp:
        json.dump(predictions, fp, ensure_ascii=False)
