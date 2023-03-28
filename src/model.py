import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from src.utils import ModelConfig

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()
        self.image_size = config.image_size
        self.model_type = config.model_type
        self.model_path = config.model_path
        self.conf = config.conf
        self.model = self.load_model()

    def load_model(self):
        if self.model_type == "v5":
            model = torch.hub.load('ultralytics/yolov5', "custom", path=self.model_path, _verbose=False)
        model.conf = self.conf
        return model
    
    def forward(self, image):
        predictions = self.model(image, size=self.image_size)
        predictions = predictions.pandas().xyxy[0].sort_values('xmax', ascending=False).to_dict(orient="records")
        outputs = []
        for pred in predictions:
            outputs.append({
                'bbox': [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']],
                'score': pred['confidence']
            })
        return outputs