import pandas as pd
import pickle
from PIL import Image
import os
import argparse
import glob
Image.MAX_IMAGE_PIXELS = None

# df = pd.read_csv("../ship-detection/.extras/train.csv")
# data_list = df.to_dict('records')
# dict = {}
# for data in data_list:
#     if data['id'] not in dict:
#         dict[data['id']] = []
#     dict[data['id']].append([data['xmin'], data['ymin'], data['xmax'], data['ymax']])

# with open('anno.pkl', 'wb') as f:
#     pickle.dump(dict, f)

STRIDE = 1500
IMAGE_SIZE = [2048, 2048]

def save_txt(data, save_path):
    with open(save_path, 'w') as fp:
        for line in data:
            fp.write(" ".join(map(str, line)) + "\n")

def get_subimage(image_path, anno, image_save_dir, label_save_dir):
    image = Image.open(image_path)
    base_f = os.path.basename(image_path)

    width, height = image.size
    limit_h = STRIDE * ((height - IMAGE_SIZE[0] - 1) // STRIDE + 1) + 1
    limit_w = STRIDE * ((width - IMAGE_SIZE[1] -1) // STRIDE + 1) + 1
    cnt = 0
    for i in range(0, limit_h, STRIDE):
        for j in range(0, limit_w, STRIDE):
            x1, y1, x2, y2 = j, i, j + IMAGE_SIZE[1], i + IMAGE_SIZE[0]
            if x2 > width:
                x1, x2 = max(width - IMAGE_SIZE[1], 0), width
            if y2 > height:
                y1, y2 = max(height - IMAGE_SIZE[0], 0), height

            sub_image = image.crop((x1, y1, x2, y2))
            data = []
            for bbox in anno[base_f]:
                minX, minY, maxX, maxY = bbox
                if minX >= x1 and maxX <= x2 and minY >= y1 and maxY <= y2:
                    centerX = (minX + maxX - 2*x1) / 2
                    centerY = (minY + maxY - 2*y1) / 2
                    image_w = x2 - x1
                    image_h = y2 - y1
                    label = 0
                    data.append([int(0), centerX/image_w, centerY/image_h, (maxX-minX)/image_w, (maxY-minY)/image_h])
            if len(data) > 0:
                sub_image.save(os.path.join(image_save_dir, base_f[:-4] + f"_{cnt}.png"))
                save_txt(data, os.path.join(label_save_dir, base_f[:-4] + f"_{cnt}.txt"))
                cnt += 1

def main(args):
    img_save_dir = os.path.join(args.save_dir, "images")
    label_save_dir = os.path.join(args.save_dir, "labels")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    anno = pickle.load(open('anno.pkl', 'rb'))
    image_list = glob.glob(f"{args.data_dir}/**/*.png", recursive=True)
    for image_path in image_list:
        get_subimage(image_path, anno, img_save_dir, label_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/raw_images/eval")
    parser.add_argument("--save_dir", default="../data/eval")
    args = parser.parse_args()

    main(args)