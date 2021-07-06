from math import sqrt
from collections import defaultdict
from itertools import product

import torch
import torch.nn as nn

from methods.instance_segmentation.yolact.parameters import *


def make_net():
    net = []
    net.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
    net.append(nn.ReLU(inplace=True))

    return nn.Sequential(*(net)), 256


class PriorData:
    _tmp_img_w, _tmp_img_h = MAX_SIZE, MAX_SIZE
    prior_cache = defaultdict(lambda: None)


class PredictionModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=1024,
                 num_classes=31,
                 num_mask=32,
                 aspect_ratios=[[1]],
                 scales=[1],
                 parent=None):

        super().__init__()

        # Set dimension parameters
        self.num_anchor = sum(len(x)*len(scales) for x in aspect_ratios)
        self.num_classes = num_classes
        self.num_mask = num_mask

        # Set anchor info
        self.aspect_ratios = aspect_ratios
        self.scales = scales

        # Init instance
        self.parent = [parent]
        self.last_img_size = None
        self.priors = None

        if parent is None:
            self.upfeature = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                           nn.ReLU(inplace=True))

            self.bbox_layer = nn.Conv2d(out_channels, self.num_anchor * 4,                kernel_size=3, padding=1)
            self.conf_layer = nn.Conv2d(out_channels, self.num_anchor * self.num_classes, kernel_size=3, padding=1)
            self.mask_layer = nn.Conv2d(out_channels, self.num_anchor * self.num_mask,    kernel_size=3, padding=1)

    def forward(self, x):
        src = self if self.parent[0] is None else self.parent[0]
        conv_h, conv_w = x.size(2), x.size(3)

        x = src.upfeature(x)
        bbox = src.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        mask = src.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_mask)
        mask = torch.tanh(mask)

        priors = self.make_priors(conv_h, conv_w, x.device)

        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}
        return preds

    # 아마도 기본 anchor box를 생성하는 듯 (이미지 크기가 달라지면 달라짐)
    def make_priors(self, conv_h, conv_w, device):
        size = (conv_h, conv_w)

        if self.last_img_size != (PriorData._tmp_img_w, PriorData._tmp_img_h):
            prior_data = []
            for j, i in product(range(conv_h), range(conv_w)):
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for ars in self.aspect_ratios:
                    for scale in self.scales:
                        for ar in ars:
                            ar = sqrt(ar)

                            h = w = scale * ar / MAX_SIZE
                            # h = scale / ar / MAX_SIZE #
                            prior_data += [x, y, w, h]
            self.priors = torch.Tensor(prior_data, device=device).view(-1, 4).detach()
            self.priors.requires_grad = False
            self.last_img_size = (PriorData._tmp_img_w, PriorData._tmp_img_h)
            self.last_conv_size = (conv_w, conv_h)
            PriorData.prior_cache[size] = None

        elif self.priors.device != device:
            if PriorData.prior_cache[size] is None:
                PriorData.prior_cache[size] = {}

            if device not in PriorData.prior_cache[size]:
                PriorData.prior_cache[size][device] = self.priors.to(device)

            self.priors = PriorData.prior_cache[size][device]

        return self.priors


def make_predict_head(num_classes, num_mask, selected_layers, src_channels):
    layers = nn.ModuleList()
    scales = [24, 48, 96, 192, 384]

    for idx, layer_idx in enumerate(selected_layers):

        if idx > 0:
            parent = layers[0]
        else:
            parent = None

        pred = PredictionModule(src_channels[layer_idx],
                                src_channels[layer_idx],
                                num_classes=num_classes,
                                num_mask=num_mask,
                                aspect_ratios=[ANCHOR_ASPECT_RATIO],
                                scales=[scales[idx]],
                                parent=parent)
        layers.append(pred)
    return layers

