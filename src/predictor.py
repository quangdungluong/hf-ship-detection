import os
import sys
sys.path.insert(0, os.getcwd())

from src.model import Model
from src.wbf import wbf
from src.utils import PredictorConfig, ModelConfig

class Predictor:
    def __init__(self, predictor_config: PredictorConfig, model_config: ModelConfig):
       self.device = predictor_config.device
       self.model = Model(config=model_config)
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
                outputs = self.model(sub_image)
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