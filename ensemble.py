import torch.nn as nn
from src.utils import PredictorConfig, ModelConfig, read_image, load_ensemble_config, read_yaml
from typing import List
from src.predictor import Predictor
from src.wbf import ensemble_wbf
import argparse
import glob
import os
from tqdm.auto import tqdm
import pandas as pd
import json

class Ensemble:
    def __init__(self, predictor_config: PredictorConfig, list_model_config: List[ModelConfig]):
        self.list_predictor = [Predictor(predictor_config, model_config) for model_config in list_model_config]
        self.thres = predictor_config.wbf_thres

    def detect(self, image):
        list_predictions = [predictor.detect(image) for predictor in self.list_predictor]
        height, width, _ = image.shape
        final_results = ensemble_wbf(list_predictions, self.thres, height, width)
        return final_results
    
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
    ensemble = Ensemble(list_model_config=list_model_config, predictor_config=predictor_config)

    submission = {"id": [], "label": []}
    images_list = glob.glob(f"{args.image_dir}/**/*.png", recursive=True)
    for image_path in tqdm(images_list):
        base_f = os.path.basename(image_path)
        image = read_image(image_path)
        predictions = ensemble.detect(image)

        results_str = []
        for pred in predictions:
            result = f"{pred['score']} {pred['bbox'][0]} {pred['bbox'][1]} {pred['bbox'][2]} {pred['bbox'][3]}"
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