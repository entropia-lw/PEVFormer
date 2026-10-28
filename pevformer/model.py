import torch
from torch import nn
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import ImageList
from .head import PolygonHead

@META_ARCH_REGISTRY.register()
class PEVFormer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_proposals = cfg.MODEL.PEVFormer.NUM_PROPOSALS
        self.head = PolygonHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        self.size_divisibility = self.backbone.size_divisibility
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        images, images_whwh = self.preprocess_image(batched_inputs)
        src = self.backbone(images.tensor)
        features = []
        for f in self.in_features:
            features.append(src[f])
        if 'instances' in batched_inputs[0] and hasattr(batched_inputs[0]['instances'], 'gt_boxes'):
            gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
            proposal_boxes, box_lengths = pad_gt_boxes(gt_instances, self.num_proposals, self.device)
        else:
            proposal_boxes = []
            for bi in batched_inputs:
                h, w = bi['image'].shape[-2:]
                boxes = torch.zeros(self.num_proposals, 4, device=self.device)
                boxes[:] = torch.tensor([0, 0, w - 1, h - 1], dtype=torch.float32, device=self.device)
                proposal_boxes.append(boxes)
            proposal_boxes = torch.stack(proposal_boxes)
        outputs = self.head(features, proposal_boxes)
        return [outputs]

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image'].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        images_whwh = []
        for bi in batched_inputs:
            h, w = bi['image'].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)
        return (images, images_whwh)

def pad_gt_boxes(gt_instances, num_proposals, device):
    proposal_boxes = []
    box_lengths = []
    for gt_inst in gt_instances:
        gt_boxes = gt_inst.gt_boxes.tensor
        box_lengths.append(len(gt_boxes))
        boxes_pad = torch.zeros([num_proposals, 4], device=device)
        boxes_pad[:len(gt_boxes)] = gt_boxes
        proposal_boxes.append(boxes_pad)
    proposal_boxes = torch.stack(proposal_boxes).view(-1, num_proposals, 4)
    box_lengths = torch.tensor(box_lengths, device=device)
    return (proposal_boxes, box_lengths)
