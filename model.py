"""
Model module that defines Mask R-CNN with Swin Transformer backbone, FPN and PAN neck.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models._utils import IntermediateLayerGetter


class PathAggregationNetwork(nn.Module):
    """Path Aggregation Network (PAN) neck module."""

    def __init__(self, channels):
        super().__init__()
        c_2, c_3, c_4, c_5 = channels

        # Top-down
        self.reduce_c5 = nn.Conv2d(c_5, c_4, 1)
        self.reduce_c4 = nn.Conv2d(c_4, c_3, 1)
        self.reduce_c3 = nn.Conv2d(c_3, c_2, 1)

        self.up_c4 = nn.Conv2d(c_4 + c_4, c_4, 3, padding=1)
        self.up_c3 = nn.Conv2d(c_3 + c_3, c_3, 3, padding=1)
        self.up_c2 = nn.Conv2d(c_2 + c_2, c_2, 3, padding=1)

        # Bottom-up
        self.down_c3 = nn.Conv2d(c_2, c_3, 3, stride=2, padding=1)
        self.fuse_c3 = nn.Conv2d(c_3 + c_3, c_3, 3, padding=1)

        self.down_c4 = nn.Conv2d(c_3, c_4, 3, stride=2, padding=1)
        self.fuse_c4 = nn.Conv2d(c_4 + c_4, c_4, 3, padding=1)

        self.down_c5 = nn.Conv2d(c_4, c_5, 3, stride=2, padding=1)
        self.fuse_c5 = nn.Conv2d(c_5 + c_5, c_5, 3, padding=1)

    def forward(self, features):
        """Forward pass through PAN."""
        c_2, c_3, c_4, c_5 = features['0'], features['1'], features['2'], features['3']

        # Top-down
        p_5 = self.reduce_c5(c_5)
        p_4 = self.up_c4(
            torch.cat([F.interpolate(p_5, scale_factor=2, mode='nearest'), c_4], dim=1))
        p_4_reduced = self.reduce_c4(p_4)
        p_3 = self.up_c3(torch.cat(
            [F.interpolate(p_4_reduced, scale_factor=2, mode='nearest'), c_3], dim=1))
        p_3_reduced = self.reduce_c3(p_3)
        p_2 = self.up_c2(torch.cat(
            [F.interpolate(p_3_reduced, scale_factor=2, mode='nearest'), c_2], dim=1))

        # Bottom-up
        p_3_out = self.fuse_c3(torch.cat([self.down_c3(p_2), p_3], dim=1))
        p_4_out = self.fuse_c4(torch.cat([self.down_c4(p_3_out), p_4], dim=1))
        p_5_out = self.fuse_c5(torch.cat([self.down_c5(p_4_out), p_5], dim=1))

        return {
            '0': p_2,
            '1': p_3_out,
            '2': p_4_out,
            '3': p_5_out
        }


class SwinBackboneWithFPN(nn.Module):
    """Swin Transformer backbone with FPN."""

    def __init__(self):
        super().__init__()
        self.backbone = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT).features
        self.body = IntermediateLayerGetter(
            self.backbone,
            OrderedDict({
                '1': '0',
                '3': '1',
                '5': '2',
                '7': '3'
            })
        )

        self.out_channels = 256
        self.in_channels_list = [96, 192, 384, 768]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

    def forward(self, input_tensor):
        """Forward pass through Swin + FPN."""
        x = self.body(input_tensor)
        x = {k: v.permute(0, 3, 1, 2) for k, v in x.items()}
        return self.fpn(x)


class CombinedBackbone(nn.Module):
    """Combines Swin + FPN + PAN into a single backbone."""

    def __init__(self, backbone_fpn, panet):
        super().__init__()
        self.backbone_fpn = backbone_fpn
        self.panet = panet
        self.out_channels = 256

    def forward(self, input_tensor):
        features = self.backbone_fpn(input_tensor)
        return self.panet(features)


def build_swin_maskrcnn(num_classes, device):
    """Returns the full Mask R-CNN model with Swin+FPN+PAN."""
    swin_backbone = SwinBackboneWithFPN()
    panet = PathAggregationNetwork([swin_backbone.out_channels] * 4)
    combined_backbone = CombinedBackbone(swin_backbone, panet)

    anchor_sizes = ((4,), (8,), (16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * 4

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    num_anchors = anchor_generator.num_anchors_per_location()[0]

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    model = MaskRCNN(
        backbone=combined_backbone,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        num_classes=num_classes
    )

    model.roi_heads.positive_fraction = 0.25

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.rpn.head = RPNHead(in_channels=256, num_anchors=num_anchors)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes=num_classes
    )

    return model.to(device)
