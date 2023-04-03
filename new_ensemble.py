import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
import glob
import json
from typing import List

import pandas as pd
from src.model import Model
from src.utils import (ModelConfig, PredictorConfig, load_ensemble_config,
                       read_image, read_yaml)
from src.wbf import wbf
from tqdm.auto import tqdm


class EnsemblePredictor:
    def __init__(self, predictor_config: PredictorConfig, list_model_config: List[ModelConfig]):
       self.device = predictor_config.device
       self.list_model = [Model(model_config) for model_config in list_model_config]
       self.stride = predictor_config.stride
       self.image_size = predictor_config.image_size
       self.wbf_thres = predictor_config.wbf_thres
        
    def detect_on_sub_image(self, image):
        height, width, _ = image.shape
        limit_h = self.stride * ((height - self.image_size[0] - 1) // self.stride + 1) + 1
        limit_w = self.stride * ((width - self.image_size[1] - 1) // self.stride + 1) + 1

        all_predictions = []
        for i in range(0, limit_h, self.stride):
            for j in range(0, limit_w, self.stride):
                x1, y1, x2, y2 = j, i, j + self.image_size[1], i + self.image_size[0]
                if x2 > width:
                    x1, x2 = max(width - self.image_size[1], 0), width
                if y2 > height:
                    y1, y2 = max(height - self.image_size[0], 0), height

                sub_image = image[y1:y2, x1:x2]
                outputs = []
                for model in self.list_model:
                    outputs.extend(model(sub_image))
                for output in outputs:
                    output['bbox'][0] += x1
                    output['bbox'][1] += y1
                    output['bbox'][2] += x1
                    output['bbox'][3] += y1
                    all_predictions.append(output)
        return all_predictions
    
    def wbf_on_image(self, all_predictions, height, width):
        outputs = wbf(all_predictions, thres=self.wbf_thres, height=height, width=width)
        return outputs
    
    def detect(self, image):
        height, width, _ = image.shape
        all_predictions = self.detect_on_sub_image(image)
        all_predictions = self.wbf_on_image(all_predictions=all_predictions, height=height, width=width)
        return all_predictions
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--config_dict", type=json.loads)
    """
    {"v5": {"model_list": [v1, v2, v3], "config_path": ""}, "v7": {}}
    """    
    parser.add_argument("--image_dir", type=str, default="../ship-detection/test")
    args = parser.parse_args()

    config_path = "./config/config.yaml"
    cfg = read_yaml(config_path)
    list_model_config, predictor_config = load_ensemble_config(cfg, args.config_dict)
    ensemble = EnsemblePredictor(list_model_config=list_model_config, predictor_config=predictor_config)

    submission = {"id": [], "label": []}
    images_list = glob.glob(f"{args.image_dir}/**/*.png", recursive=True)
    for image_path in tqdm(images_list):
        base_f = os.path.basename(image_path)
        image = read_image(image_path)
        predictions = ensemble.detect(image)

        results_str = []
        for pred in predictions:
            result = f"{pred['score']:.3f} {pred['bbox'][0]:.1f} {pred['bbox'][1]:.1f} {pred['bbox'][2]:.1f} {pred['bbox'][3]:.1f}"
            results_str.append(result)

        if len(results_str) > 1:
            results_str = ', '.join(results_str)
        elif len(results_str) == 1:
            results_str = results_str[0]
        else:
            results_str = "0 0 0 10 10"
        submission['id'].append(base_f)
        submission['label'].append(results_str)

    df = pd.DataFrame.from_dict(submission)
    df = df.sort_values(by=['id'])
    df.to_csv("./submission.csv", index=False)