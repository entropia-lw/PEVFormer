import copy
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, PolygonMasks
__all__ = ['PEVFormerDatasetMapper']

class PEVFormerDatasetMapper:

    def __init__(self, cfg, is_train=False):
        self.tfm_gens = []
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.num_corners = cfg.MODEL.PEVFormer.NUM_CORNERS

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if 'annotations' in dataset_dict:
            annos = [utils.transform_instance_annotations(obj, transforms, image_shape) for obj in dataset_dict.pop('annotations') if obj.get('iscrowd', 0) == 0]
            instances = annotations_to_instances(annos, image_shape, self.is_train, self.num_corners)
            dataset_dict['instances'] = detectron2_filter_empty_instances(instances)
        return dataset_dict

def annotations_to_instances(annos, image_size, is_train, num_corners):
    boxes = np.stack([BoxMode.convert(obj['bbox'], obj['bbox_mode'], BoxMode.XYXY_ABS) for obj in annos]) if len(annos) else np.zeros((0, 4))
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    if len(annos):
        classes = np.stack([int(obj['category_id']) for obj in annos])
        target.gt_classes = torch.tensor(classes, dtype=torch.int64)
    else:
        target.gt_classes = torch.zeros(torch.Size([0]), dtype=torch.int32)
    if len(annos) and 'segmentation' in annos[0]:
        segms = [obj['segmentation'] for obj in annos]
        masks = PolygonMasks(segms)
        target.gt_masks = masks
    return target

def detectron2_filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-05, return_mask=False):
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has('gt_masks') and by_mask:
        r.append(instances.gt_masks.nonempty())
    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return (instances[m], m)
    return instances[m]
