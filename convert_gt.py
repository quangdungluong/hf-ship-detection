import pickle
import json
import os
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

labels = pickle.load(open("./data/anno.pkl", 'rb'))
val_image_list = glob.glob("../data/raw_images/eval/**/*.png", recursive=True)

id = 0
def run():
    images = []
    annotations = []

    def read_annotation(data, img_id):
        global id
        annotation = []
        for box in data:
            x0, y0, x1, y1 = box
            segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            bbox = [x0, y0, x1-x0, y1-y0]
            area = (x1-x0)*(y1-y0)
            annotation.append({
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': 1,
                'id': id
            })
            id += 1
        return annotation

    for image_path in val_image_list:
        base_f = os.path.basename(image_path)
        image = Image.open(image_path)
        width, height = image.size
        images.append({
            'date_captured': '2023',
            'file_name': base_f,
            'id': int(base_f.split('.')[0]),
            'height': height,
            'width': width
        })
        annotation_data = labels[base_f]
        new_anno = read_annotation(annotation_data, int(base_f.split('.')[0]))
        if len(new_anno) > 0:
            annotations.extend(new_anno)

    info = {'year': int(2023),
            'version': '1.0',
            'description': 'For object detection',
            'date_created': 2023,}
    licenses = [{'id': 1,
                'name': 'Apache License v2.0'}]
    json_data = {
        'info': info,
        'images': images,
        'license': licenses,
        'type': 'instances',
        'annotations': annotations,
        'categories': [{'supercategory': 'ship',
                        'id': 1,
                        'name': 'ship',}]
    }
    with open('./data/gt_val.json', 'w', encoding='utf-8') as fp:
        json.dump(json_data, fp, ensure_ascii=False)