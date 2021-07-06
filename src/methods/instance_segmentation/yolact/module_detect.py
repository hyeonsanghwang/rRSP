import torch

from methods.instance_segmentation.yolact.box_utils import decode, jaccard


class Detect(object):
    MAX_NUM_DETECTION = 100

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.conf_thresh = torch.cuda.FloatTensor([conf_thresh])


        self.nms_thresh = nms_thresh

    def __call__(self, predictions, net):
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        prior_data = predictions['priors']
        proto_data = predictions['proto']

        out = []
        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()

        for batch_idx in range(batch_size):
            decoded_boxes = decode(loc_data[batch_idx], prior_data)

            result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data)
            if result is not None and proto_data is not None:
                result['proto'] = proto_data[batch_idx]
            out.append({'detection': result, 'net': net})
        return out

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)

        # Processing
        keep = conf_scores > self.conf_thresh
        masks = mask_data[batch_idx, keep, :]
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]

        if scores.size(1) == 0:
            return None

        boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def fast_nms(self, boxes, masks, scores, iou_threshold, top_k):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        keep = (iou_max <= iou_threshold)

        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        scores, idx = scores.sort(0, descending=True)
        idx = idx[:self.MAX_NUM_DETECTION]
        scores = scores[:self.MAX_NUM_DETECTION]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores



