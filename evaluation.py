from mapcalc import calculate_map, calculate_map_range
import glob
import pickle
import os

ground_truth = {"boxes": [],
                "labels": []}
result_dict = {"boxes": [],
               "labels": [],
               "scores": []}

eval_list = glob.glob("../data/raw_images/eval/**/*.png", recursive=True)
anno = pickle.load(open('anno.pkl', 'rb'))

for image_path in eval_list:
    base_f = os.path.basename(image_path)
    for bbox in anno[base_f]:
        ground_truth['boxes'].append(bbox)
        ground_truth["labels"].append(0)

        result_dict["boxes"].append(bbox)
        result_dict['labels'].append(0)
        result_dict['scores'].append(0.7)

# calculates the mAP for an IOU threshold of 0.5
print(calculate_map(ground_truth, result_dict, 0.5))

# calculates the mAP average for the IOU thresholds 0.05, 0.1, 0.15, ..., 0.90, 0.95.
print(calculate_map_range(ground_truth, result_dict, 0.05, 0.95, 0.05))