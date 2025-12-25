import json
import random
import os
import argparse

parser = argparse.ArgumentParser(description='Create a subset of COCO validation annotations.')
parser.add_argument('--val_json_path', type=str, required=True, help='Path to the original COCO validation JSON file.')
parser.add_argument('--output_json_path', type=str, required=True, help='Path to save the subset JSON file.')
parser.add_argument('--subset_size', type=int, default=50, help='Number of images to include in the subset.')
args = parser.parse_args()

val_json_path = args.val_json_path
output_json_path = args.output_json_path
subset_size = args.subset_size

with open(val_json_path, 'r') as f:
    coco_data = json.load(f)

original_images = coco_data['images']
random.seed(42)
subset_images = random.sample(original_images, subset_size)

subset_img_ids = {img['id'] for img in subset_images}

subset_annotations = [
    ann for ann in coco_data['annotations']
    if ann['image_id'] in subset_img_ids
]

subset_data = {
    'info': coco_data['info'],
    'licenses': coco_data['licenses'],
    'images': subset_images,
    'annotations': subset_annotations,
    'categories': coco_data['categories']
}

with open(output_json_path, 'w') as f:
    json.dump(subset_data, f)
