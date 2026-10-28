import argparse
import os
from eval_coco import eval_coco


def main():
    parser = argparse.ArgumentParser(description="PEVFormer cloud-compatible COCO segmentation evaluation")
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--dt-file", required=True)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--eval-type", choices=["coco_iou"], default="coco_iou")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    eval_coco(args.gt_file, args.dt_file, args.output)


if __name__ == "__main__":
    main()
