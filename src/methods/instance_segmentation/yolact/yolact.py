import torch

import torch.nn as nn
import torch.nn.functional as F

from methods.instance_segmentation.yolact.parameters import *
from methods.instance_segmentation.yolact.module_backbone import ResNetBackbone
from methods.instance_segmentation.yolact.module_protonet import make_protonet
from methods.instance_segmentation.yolact.module_fpn import FPN
from methods.instance_segmentation.yolact.module_predict import make_predict_head, PriorData
from methods.instance_segmentation.yolact.module_detect import Detect


class Yolact(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone network (ResNet-101 -> ResNet-50)
        self.backbone = ResNetBackbone(LAYERS[backbone_type])

        # ProtoNet
        in_channels = 256
        self.proto_net, mask_dim = make_protonet(in_channels)

        # FPN
        self.backbone_selected_layers = PREDICTION_SELECTED_LAYERS
        src_channels = self.backbone.channels
        target_channels = [src_channels[i] for i in self.backbone_selected_layers]

        self.fpn = FPN(target_channels, NUM_DOWNSAMPLE)
        self.fpn_selected_layers = list(range(len(self.backbone_selected_layers) + NUM_DOWNSAMPLE))
        src_channels = [256] * len(self.fpn_selected_layers)

        # Predict layer
        self.prediction_layers = make_predict_head(NUM_CLASSES, mask_dim, self.fpn_selected_layers, src_channels)

        # Semantic segmentation convolution (for training)
        self.semantic_seg_conv = nn.Conv2d(src_channels[0], NUM_CLASSES-1, kernel_size=1)

        # Detection (for testing)
        nms_top_k = 200
        self.detect = Detect(NUM_CLASSES,
                             bkg_label=0,
                             top_k=nms_top_k,
                             conf_thresh=CONF_THRESH,
                             nms_thresh=NMS_THRESH)

    def forward(self, x):
        _, _, img_h, img_w = x.size()
        PriorData._tmp_img_h = img_h
        PriorData._tmp_img_w = img_w

        # Backbone
        outs = self.backbone(x)

        # FPN
        outs = [outs[i] for i in self.backbone_selected_layers]
        outs = self.fpn(outs)

        # ProtoNet
        proto_x = outs[0]
        proto_out = self.proto_net(proto_x)
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        # Predict
        pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}

        for i, (idx, pred_layer) in enumerate(zip(self.fpn_selected_layers, self.prediction_layers)):
            pred_x = outs[idx]
            if pred_layer is not self.prediction_layers[0]:
                pred_layer.parent = [self.prediction_layers[0]]
            pred = pred_layer(pred_x)
            for k, v in pred.items():
                pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)
        pred_outs['proto'] = proto_out

        if self.training:
            pred_outs['segm'] = self.semantic_seg_conv(outs[0])
            return pred_outs
        else:
            pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            res = self.detect(pred_outs, self)
            return res

    def init_weights(self, backbone_path='weights/resnet101_reducedfc.pth'):
        self.backbone.init_backbone(backbone_path)

        for name, module in self.named_modules():
            is_conv_layer = isinstance(module, nn.Conv2d)
            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            for key in list(state_dict.keys()):
                if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                    del state_dict[key]
                elif key.startswith('fpn.lat_layers'):  # For compatibility with the original model
                    state_dict['fpn.mid_layers' + key[14:]] = state_dict[key]
                    del state_dict[key]

        self.load_state_dict(state_dict)


from torch.utils.data import DataLoader, Dataset
import numpy as np
from time import perf_counter


class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(0, 3, 1, 2)

        self.y_data = torch.LongTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.device('cuda:0')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create sample data
    sample_x = np.random.randint(256, size=(1, MAX_SIZE, MAX_SIZE, 3)) / 256.0
    sample_y = np.random.randint(2, size=(1, 1))

    # Create data loader
    data = TensorData(sample_x, sample_y)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    iter_data = iter(data_loader)

    # Craete model
    yolact = Yolact()
    yolact.init_weights()

    # Test
    images, labels = iter_data.next()
    start_t = perf_counter()
    res = yolact(images.cuda())
    print(perf_counter() - start_t)
