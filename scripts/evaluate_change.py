import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pevformer.utils import load_checkpoint

def setup_cfg(args):
    cfg = get_cfg()
    add_pevformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.CORNER_THRESHOLD = args.corner_threshold
    cfg.OUTPUT_DIRPATH = args.output
    cfg.ITER = args.iter
    cfg.IMAGE_ID = args.image_id
    cfg.freeze()
    return cfg
import json
import time
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import torch
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from pevformer import PEVFormerDatasetMapper, add_pevformer_config
import shapely.geometry
from detectron2.config import get_cfg
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Unified evaluation script for PEVFormer models')
    parser.add_argument('--config-file', default='configs/pevformer.res50.crowdai.sm.yaml', metavar='FILE', help='Path to config file')
    parser.add_argument('--output', help='Directory to save output visualizations')
    parser.add_argument('--gt-path', help='Directory of the annotation file')
    parser.add_argument('--corner-threshold', type=float, default=0.4, help='Threshold for vertex predictions to be classified as corners')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--iter', type=int, default=2842656, help='Checkpoint iteration number')
    parser.add_argument('--image-id', type=int, default=1, help='ID of the input image')
    parser.add_argument('--train-json', help='Path to the COCO-format annotation file')
    parser.add_argument('--train-path', help='Path to the images directory')
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line 'KEY VALUE' pairs")
    return parser

def single_annotation(image_id, poly, bbox, score):
    return {'image_id': int(image_id), 'category_id': 100, 'score': score, 'segmentation': poly, 'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]}

def prediction(cfg):
    mapper = PEVFormerDatasetMapper(cfg, is_train=False)
    dataloader = build_detection_test_loader(DatasetCatalog.get('aicrowd_test'), mapper=mapper)
    test_iterator = tqdm(dataloader)
    model = build_model(cfg)
    model.eval()
    model.load_state_dict(load_checkpoint(cfg.MODEL.WEIGHTS)['model'], strict=True)
    has_cfe_module = any(('contour_enhancer' in name for name, _ in model.named_modules()))
    print(f'[INFO] Model architecture has contour_enhancer module: {has_cfe_module}')
    print(f'[INFO] Config USE_CONTOUR_ENHANCER: {cfg.MODEL.PEVFormer.USE_BAFE}')
    speed = []
    predictions = []
    total_rois = 0
    total_valid_polys = 0
    total_filtered_by_area = 0
    total_filtered_by_vertices = 0
    for i_batch, batched_inputs in enumerate(test_iterator):
        h, w = (batched_inputs[0]['height'], batched_inputs[0]['width'])
        instances = batched_inputs[0]['instances']
        gt_boxes = instances.gt_boxes.tensor
        num_gt_boxes = gt_boxes.shape[0]
        with torch.no_grad():
            t0 = time.time()
            outputs = model(batched_inputs)
            if isinstance(outputs, list):
                if i_batch == 0:
                    print('=' * 80)
                    print('[INFO] Model output format detected: LIST')
                    print(f'[INFO] This appears to be a PEVFormer model')
                    print(f'[INFO] List length: {len(outputs)}')
                    print(f'[INFO] Unpacking first element...')
                    print('=' * 80)
                outputs = outputs[0]
            elif isinstance(outputs, dict):
                if i_batch == 0:
                    print('=' * 80)
                    print('[INFO] Model output format detected: DICT')
                    print(f'[INFO] This appears to be a Baseline model')
                    print(f'[INFO] No unpacking needed')
                    print('=' * 80)
            else:
                raise TypeError(f"\n{'=' * 80}\n[ERROR] Model returned unexpected output type: {type(outputs)}\nExpected: dict or list[dict]\nPlease check the model's forward() function\n{'=' * 80}\n")
            t1 = time.time()
        pred_logits = outputs['pred_logits'][0]
        pred_coords = outputs['pred_coords'][0]
        fg_mask = torch.sigmoid(pred_logits) > cfg.CORNER_THRESHOLD
        polys_image = []
        scores_image = []
        boxes_image = []
        total_rois += num_gt_boxes
        for j in range(num_gt_boxes):
            fg_mask_per_poly = fg_mask[j]
            valid_coords_per_poly = pred_coords[j][fg_mask_per_poly]
            valid_scores_per_poly = torch.sigmoid(pred_logits[j])[fg_mask_per_poly]
            if len(valid_coords_per_poly) > 0:
                coords = valid_coords_per_poly.cpu().numpy()
                if len(coords) >= 3 and shapely.geometry.Polygon(coords).area >= 10:
                    polys_image.append(coords)
                    scores_image.append(torch.mean(valid_scores_per_poly).cpu().numpy())
                    boxes_image.append(np.array([np.min(coords[:, 0]), np.min(coords[:, 1]), np.max(coords[:, 0]), np.max(coords[:, 1])]))
                    total_valid_polys += 1
                else:
                    total_filtered_by_area += 1
        image_id = batched_inputs[0]['image_id']
        for j, poly in enumerate(polys_image):
            box = list(np.array(boxes_image[j]).flatten().astype(np.float64))
            score = float(scores_image[j])
            poly = list(np.array(poly).flatten().astype(np.float64))
            if len(poly) > 4:
                predictions.append(single_annotation(image_id, [poly], box, score))
            else:
                total_filtered_by_vertices += 1
        speed.append(t1 - t0)
    print('\n' + '=' * 80)
    print('Evaluation Summary:')
    print(f'  Total images processed: {len(speed)}')
    print(f'  Total RoIs: {total_rois}')
    print(f'  Valid polygons (area >= 10): {total_valid_polys}')
    print(f'  Filtered by area (<10 or <3 points): {total_filtered_by_area}')
    print(f'  Filtered by vertex count (<=2 vertices): {total_filtered_by_vertices}')
    print(f'  Final predictions: {len(predictions)}')
    print(f'  Average speed: {np.mean(speed):.4f} s/image')
    print('=' * 80)
    output_path = os.path.join(cfg.OUTPUT_DIRPATH, f'predictions_{int(cfg.ITER)}.json')
    with open(output_path, 'w') as fp:
        json.dump(predictions, fp)
    print(f'\n[SUCCESS] Predictions saved to: {output_path}')

def register_my_dataset(dataset_name='aicrowd_test', TEST_JSON='../data/crowdai/val/annotation_sparsercnn_swinb_1930059_sm.json', TEST_PATH='../data/crowdai/val/images'):
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(TEST_JSON, TEST_PATH, dataset_name))
    MetadataCatalog.get(dataset_name).set(json_file=TEST_JSON, image_root=TEST_PATH)
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = get_parser().parse_args()
    setup_logger(name='fvcore')
    logger = setup_logger()
    logger.info('Arguments: ' + str(args))
    os.makedirs(args.output, exist_ok=True)
    cfg = setup_cfg(args)
    register_my_dataset(dataset_name='aicrowd_test', TEST_JSON=args.train_json, TEST_PATH=args.train_path)
    print('\n' + '=' * 80)
    print('Starting unified evaluation (evaluate_change.py)')
    print(f'  Config file             : {args.config_file}')
    print(f'  Output directory         : {args.output}')
    print(f'  Checkpoint iter          : {args.iter}')
    print(f'  Corner threshold         : {args.corner_threshold}')
    print(f'  USE_CONTOUR_ENHANCER     : {cfg.MODEL.PEVFormer.USE_BAFE}')
    print(f'  (auto-detected from checkpoint)')
    print('=' * 80 + '\n')
    prediction(cfg)
