import json

import numpy as np
import yaml
from PIL import Image
from dataclasses import dataclass

Image.MAX_IMAGE_PIXELS = None

def read_image(image_path):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    return image

def get_image_shape(image):
    height, width, _ = image.shape
    return height, width

def read_json(file_path):
    file = open(file_path, encoding='utf-8')
    data = json.loads(file.read())
    return data

def read_yaml(file_path):
    file = open(file_path, encoding='utf-8')
    data = yaml.safe_load(file.read())
    return data

@dataclass
class ModelConfig:
    image_size: int
    model_type: str
    model_path: str
    conf: float

@dataclass
class PredictorConfig:
    device: str
    stride: int
    image_size: list
    wbf_thres: float

def load_config(cfg):
    detail_cfg = read_yaml(cfg['model']['config_path'])
    model_type = cfg['model']['type']
    model_version = cfg['model']['version']

    model_config = ModelConfig(
        image_size=detail_cfg[model_version]['img_size'],
        model_type=model_type,
        model_path=detail_cfg[model_version]['model_path'],
        conf=detail_cfg[model_version]['conf']
    )

    predictor_config = PredictorConfig(
        device=cfg['predictor']['device'],
        stride=cfg['predictor']['stride'],
        image_size=cfg['predictor']['image_size'],
        wbf_thres=cfg['predictor']['wbf_thres']
    )
    return model_config, predictor_config

def load_ensemble_config(cfg, list_model):
    detail_cfg = read_yaml(cfg['model']['config_path'])
    model_type = cfg['model']['type']

    list_model_config = []

    for model_version in list_model:
        list_model_config.append(ModelConfig(
                                image_size=detail_cfg[model_version]['img_size'],
                                model_type=model_type,
                                model_path=detail_cfg[model_version]['model_path'],
                                conf=detail_cfg[model_version]['conf']
    ))

    predictor_config = PredictorConfig(
        device=cfg['predictor']['device'],
        stride=cfg['predictor']['stride'],
        image_size=cfg['predictor']['image_size'],
        wbf_thres=cfg['predictor']['wbf_thres']
    )
    return list_model_config, predictor_config