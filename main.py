from src.predictor import Predictor
from src.utils import load_config, read_image
import argparse
import glob
from tqdm.auto import tqdm
import os
from src.logger import initial_logger
import pandas as pd

logger = initial_logger()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./weights/test.pt")
    parser.add_argument("--image_dir", type=str, default="../ship-detection/test")
    args = parser.parse_args()

    config_path = "./config/config.yaml"
    model_cfg, predictor_cfg = load_config(config_path)
    model_cfg.model_path = args.model_path
    predictor = Predictor(predictor_config=predictor_cfg, model_config=model_cfg)

    submission = {"id": [], "label": []}
    images_list = glob.glob(f"{args.image_dir}/**/*.png", recursive=True)
    for image_path in tqdm(images_list):
        base_f = os.path.basename(image_path)
        image = read_image(image_path)
        predictions = predictor.detect(image)

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
    df.to_csv("./submission.csv", index=False)
    print(df)