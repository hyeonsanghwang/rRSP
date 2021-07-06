import cv2
import numpy as np

from collections import defaultdict

import torch
from torch.backends import cudnn

from methods.instance_segmentation.yolact.yolact import Yolact
from methods.instance_segmentation.yolact.utils import get_torso_mask
from gui.tab_segmentation import SegmentationParameter

from methods.instance_segmentation.yolact.parameters import *


class SegmentationBased:
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, params: SegmentationParameter):
        self.model_path = '../model/instance_segmentation/yolact_base_3703_799658.pth'
        self.use_gpu = True

        # Load yolact
        self._load_model()

        # Set parameters
        self.window_size = 150
        self.maximum_point = 200

        # buffers
        self.return_signal = np.array([0] * self.window_size, np.float32)
        self.signal = [0]
        self.frame = None

        self.gray = None
        self.prev_gray = None

        self.prev_points = None
        self.curr_points = None

        # Draw
        self.color_cache = defaultdict(lambda: {})
        self.draw_data = {}

        # ETC
        self.show_mode = 0

    def _load_model(self):
        with torch.no_grad():
            cudnn.fastest = True
            if torch.cuda.is_available() and self.use_gpu:
                torch.device('cuda:0')
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.device('cpu')
                torch.set_default_tensor_type('torch.FloatTensor')

            self.model = Yolact()
            self.model.load_weights(self.model_path)
            self.model.eval()
            if self.use_gpu:
                self.model.cuda()

    def reset(self):
        pass

    def get_show_frame(self):
        if self.show_mode == SegmentationParameter.SHOW_MODE_ORIGINAL:
            return self.frame
        elif self.show_mode == SegmentationParameter.SHOW_MODE_SEGMENTATION:
            return self._draw_segmentation_result(self.draw_data["frame"], self.draw_data["masks"], self.draw_data["classes"])
        elif self.show_mode == SegmentationParameter.SHOW_MODE_MASK:
            return (self.mask * 255).astype(np.uint8)
        elif self.show_mode == SegmentationParameter.SHOW_MODE_MOTION:
            if self.prev_points is None or self.curr_points is None:
                return self.frame
            else:
                show_frame = self.frame.copy()
                mask = np.zeros_like(show_frame)
                for i, (curr, prev) in enumerate(zip(self.curr_points, self.prev_points)):
                    x1, y1 = prev.ravel()
                    x2, y2 = curr.ravel()
                    mask = cv2.line(mask, (x2, y2), (x1, y1), (0, 255, 255), 2)
                    show_frame = cv2.circle(show_frame, (x2, y2), 3, (0, 255, 255), -1)
                return show_frame

    def get_respiration(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = self.gray
            return self.return_signal

        # Predict torso mask
        mask, self.draw_data = get_torso_mask(frame, self.model, self.use_gpu)
        if mask is None:
            return self.return_signal

        # Erode mask
        self.mask = self._get_eroded_mask(mask)

        # Calculate ROI points
        ps = self._calc_roi_points(mask)
        if ps.shape[0] == 0:
            return self.return_signal

        # Calculate optical flow
        self.prev_points, self.curr_points = self._get_optical_flow(ps)

        # Estimate motion
        movement = self._get_motion_information(self.prev_points, self.curr_points)

        # Set signal
        self.signal.append(self.signal[-1] - movement)
        if len(self.signal) > self.window_size:
            del self.signal[0]

        return np.array(self.signal, np.float32)

    def _get_eroded_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
        erode = cv2.erode(mask, kernel, anchor=(-1, -1), iterations=3)
        return erode

    def _calc_roi_points(self, mask):
        sel_pixel_num = min(int(mask.sum()), self.maximum_point)
        yp, xp = np.asarray(mask).nonzero()
        idxs = np.random.choice(xp.shape[0], sel_pixel_num, replace=False)
        xp = xp[idxs]
        yp = yp[idxs]
        return np.append(np.expand_dims(xp, -1), np.expand_dims(yp, -1), axis=-1).astype(np.float32)

    def _get_optical_flow(self, ps):
        cp, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, ps, None, **self.lk_params)
        self.prev_gray = self.gray
        pp = ps[st[:, 0] == 1]
        cp = cp[st[:, 0] == 1]
        return pp, cp

    def _get_motion_information(self, prev, curr):
        diff = curr[:, 1] - prev[:, 1]
        return diff.mean()

        diff.sort()
        diff = diff[diff.shape[0] // 3: -diff.shape[0] // 3]

        # print(np.fabs(diff).mean())
        sign_mask = diff > 0
        positive = diff[sign_mask]
        negative = diff[np.logical_not(sign_mask)]
        b = 3
        l = 0.15
        r = 1
        positive[:] = (np.power(positive, b) * np.exp(-positive / l)) / np.exp(-r / l)
        negative[:] = (np.power(negative, b) * np.exp(negative / l)) / np.exp(-r / l)

        # diff[diff < 0] = np.exp(2 * diff)
        # diff[diff > 0] = 1
        # diff[diff < 0] = -1
        diff = diff.mean()

    def _draw_segmentation_result(self, image, masks, classes):
        img_gpu = image / 255.0
        num_dets_to_consider = masks.shape[0]

        # Draw result
        def get_color(j, on_gpu=None):
            color_idx = (j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
                return self.color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.0
                    self.color_cache[on_gpu][color_idx] = color
                return color

        if num_dets_to_consider > 0:
            masks = masks[:, :, :, None]
            colors = torch.cat([get_color(classes[j], on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * MASK_ALPHA
            inv_alph_masks = masks * (-MASK_ALPHA) + 1

            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if num_dets_to_consider == 0:
            return img_numpy
        return img_numpy


if __name__ == '__main__':
    method = SegmentationBased(None)
