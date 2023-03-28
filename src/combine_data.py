import shutil
import argparse
import os
import glob
import random

def main(args):
    os.makedirs(os.path.join(args.root_dir, "train", "images"))    
    os.makedirs(os.path.join(args.root_dir, "train", "labels"))
    os.makedirs(os.path.join(args.root_dir, "val", "images"))    
    os.makedirs(os.path.join(args.root_dir, "val", "labels"))

    dirs = {"train": args.train_dir, "val": args.val_dir}
    for folder, data_dir in dirs.items():
        print(folder)
        images_list = glob.glob(f"{data_dir}/images/**/*.png", recursive=True)
        for image_path in images_list:
            shutil.copy(image_path, os.path.join(args.root_dir, folder, "images"))
            label_path = image_path.replace("images", "labels").replace("png", "txt")
            shutil.copy(label_path, os.path.join(args.root_dir, folder, "labels"))
        empty_images_list = glob.glob(f"{data_dir}/empty/**/*.png", recursive=True)
        cnt = min(len(empty_images_list), int(args.empty_ratio * len(images_list)))
        for image_path in random.choices(empty_images_list, k=cnt):
            shutil.copy(image_path, os.path.join(args.root_dir, folder, "images"))
            label_path = image_path.replace("png", "txt")
            shutil.copy(label_path, os.path.join(args.root_dir, folder, "labels"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="../data/ship-detection-v1")
    parser.add_argument("--train_dir", default="../data/train_400")
    parser.add_argument("--val_dir", default="../data/eval_400")
    parser.add_argument("--empty_ratio", default=0.1, type=float)
    args = parser.parse_args()

    main(args)