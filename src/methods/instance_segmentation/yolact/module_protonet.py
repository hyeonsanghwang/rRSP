import torch.nn as nn
import torch.nn.functional as F


class InterpolateModule(nn.Module):
    def __init__(self, *args, **kwdargs):
        super().__init__()

        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


def make_protonet(in_channels):
    # ProtoNet
    num_mask = 32
    net = []

    # 3x3 Conv x 3
    net.append(nn.Conv2d(in_channels, 256, 3, padding=1))
    net.append(nn.ReLU(inplace=True))
    net.append(nn.Conv2d(256, 256, 3, padding=1))
    net.append(nn.ReLU(inplace=True))
    net.append(nn.Conv2d(256, 256, 3, padding=1))
    net.append(nn.ReLU(inplace=True))

    # Upsampling
    net.append(InterpolateModule(scale_factor=2, mode='bilinear', align_corners=False))
    net.append(nn.ReLU(inplace=True))

    # 3x3 Conv x 1
    net.append(nn.Conv2d(256, 256, 3, padding=1))
    net.append(nn.ReLU(inplace=True))

    # 1x1 Conv for predicting mask
    net.append(nn.Conv2d(256, num_mask, 1))
    net.append(nn.ReLU(inplace=True))

    return nn.Sequential(*net), num_mask


if __name__ == '__main__':
    proto_net = make_protonet(256)
    print(proto_net)
