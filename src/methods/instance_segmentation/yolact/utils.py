import torch
import torch.nn.functional as F

from methods.instance_segmentation.yolact.box_utils import crop, sanitize_coordinates
from methods.instance_segmentation.yolact.parameters import *


# -----------------------------------------------------------------------------------------
#                                         For predict
# -----------------------------------------------------------------------------------------
class FastBaseTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
        self.std = torch.Tensor(STD).float().cuda()[None, :, None, None]

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)

        img_size = (MAX_SIZE, MAX_SIZE)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)
        img = (img - self.mean) / self.std
        img = img[:, (2, 1, 0), :, :].contiguous()
        return img

def post_process(pred, image):
    h, w, _ = image.shape

    # Post process
    pred = pred[0]
    net = pred['net']
    det = pred['detection']
    if det is None:
        return [torch.Tensor()] * 4

    if SCORE_THRESHOLD > 0:
        keep = det['score'] > SCORE_THRESHOLD
        for k in det:
            if k != 'proto':
                det[k] = det[k][keep]
        if det['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    classes = det['class']
    boxes = det['box']
    scores = det['score']
    masks = det['mask']
    proto_data = det['proto']

    masks = proto_data @ masks.t()
    masks = torch.sigmoid(masks)
    masks = crop(masks, boxes)
    masks = masks.permute(2, 0, 1).contiguous()
    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)
    masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)
    boxes = boxes.long()
    # return classes, scores, boxes, masks

    # Copy to cpu
    idx = scores.argsort(0, descending=True)[: TOP_K]
    classes = classes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()

    num_dets_to_consider = min(TOP_K, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < SCORE_THRESHOLD:
            num_dets_to_consider = j
            break
    return classes[: num_dets_to_consider], scores[: num_dets_to_consider], boxes[: num_dets_to_consider], masks[: num_dets_to_consider]

def get_torso_mask(image, model, use_gpu=False):
    if use_gpu:
        frame = torch.from_numpy(image).cuda().float()
    else:
        frame = torch.from_numpy(image).float()
    batch = FastBaseTransform()(frame.unsqueeze(0))

    preds = model(batch)
    classes, scores, boxes, masks = post_process(preds, frame)
    np_masks = masks.detach().cpu().numpy()

    try:
        torso_masks = np_masks[classes == 1]
        torso_size = torso_masks.sum(axis=(1, 2))
        torso_idx = torso_size.argmax()
        target_mask = torso_masks[torso_idx]
        sum_mask = np_masks.sum(axis=0) - target_mask
        target_mask -= sum_mask
        target_mask[target_mask < 0] = 0
        return target_mask, {"frame": frame, "masks": masks, "classes": classes}
    except:
        return None, None