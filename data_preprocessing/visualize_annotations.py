import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def draw_polygons_and_bboxes_on_image(image, annotations, line_color=(0, 255, 0), vertex_color=(0, 0, 255), vertex_thickness=5, bbox_color=(255, 0, 0)):
    for annotation in annotations:
        for polygon in annotation['segmentation']:
            polygon = np.array(polygon, np.int32).reshape((-1, 2))
            cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=2)
            for vertex in polygon:
                cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)
    return image

def process_annotations(coco_json_file, png_folder, output_folder, mask_output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)
    for image_info in tqdm(coco_data['images'], desc='Visualizing annotations'):
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(png_folder, file_name)
        if not os.path.exists(image_path):
            print(f'Image file not found: {image_path}')
            continue
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        annotations = []
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                annotations.append({'segmentation': annotation['segmentation'], 'bbox': annotation['bbox']})
        image_with_polygons_and_bboxes = draw_polygons_and_bboxes_on_image(image, annotations)
        for annotation in annotations:
            segmentation = annotation['segmentation']
            exterior = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [exterior], 255)
            if len(segmentation) > 1:
                for interior in segmentation[1:]:
                    interior_points = np.array(interior).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [interior_points], 0)
        output_image_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_image_path, image_with_polygons_and_bboxes)
        mask_output_path = os.path.join(mask_output_folder, file_name)
        cv2.imwrite(mask_output_path, mask)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw polygon contours and vertices and export building masks from COCO annotations.')
    parser.add_argument('--coco-json', required=True, help='COCO annotation JSON containing images and annotations')
    parser.add_argument('--image-dir', required=True, help='Directory containing the input images')
    parser.add_argument('--output-dir', required=True, help='Directory for polygon visualizations')
    parser.add_argument('--mask-output-dir', required=True, help='Directory for building masks')
    args = parser.parse_args()
    process_annotations(args.coco_json, args.image_dir, args.output_dir, args.mask_output_dir)
