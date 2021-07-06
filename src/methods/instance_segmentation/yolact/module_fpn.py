import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, num_downsample=2):
        super().__init__()

        mid_convs = [nn.Conv2d(x, 256, kernel_size=1) for x in reversed(in_channels)]
        self.mid_layers = nn.ModuleList(mid_convs)

        pred_convs = [nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in in_channels]
        self.pred_layers = nn.ModuleList(pred_convs)

        downsample_convs = [nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2) for _ in range(num_downsample)]
        self.downsample_layers = nn.ModuleList(downsample_convs)

    def forward(self, conv_outs):
        out = []
        x = torch.zeros(1, device=conv_outs[0].device)
        for i in range(len(conv_outs)):
            out.append(x)

        # Pyramid upsampling (P3, P4, P5)
        j = len(conv_outs)
        for layer in self.mid_layers:
            j -= 1
            if j < len(conv_outs) - 1:
                _, _, h, w = conv_outs[j].size()
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            x = x + layer(conv_outs[j])
            out[j] = x

        # Pyramid for predict (P3, P4, P5)
        j = len(conv_outs)
        for layer in self.pred_layers:
            j -= 1
            out[j] = layer(out[j])
            F.relu(out[j], inplace=True)

        # Pyramid downsampling (P6, P7)
        for layer in self.downsample_layers:
            out.append(layer(out[-1]))

        return out
