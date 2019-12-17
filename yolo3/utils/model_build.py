import torch
import numpy as np
import tqdm
from torchvision.ops.boxes import batched_nms

epsilon = 1e-16


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.

    :param prediction: (batch, height * width * num_anchors, 5 + num_classes)
    :param conf_thres:
    :param nms_thres:
    :return: detections with shape (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    prediction[..., :4] = xywh2p1p2(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none anchor are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence  (n, ) * (n, )
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        # image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), dim=1)

        keep = batched_nms(image_pred[:, :4], score, class_preds[:, 0], nms_thres)
        output[image_i] = detections[keep]

        # Perform non-maximum suppression
        # keep_boxes = []
        # while detections.size(0):
        #     large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
        #     label_match = detections[0, -1] == detections[:, -1]
        #     invalid = large_overlap & label_match
        #     weights = detections[invalid, 4:5]
        #     # Merge overlapping boxes by order of confidence
        #     detections[0, :4] = (weights / weights.sum() * detections[invalid, :4]).sum(0)
        #     keep_boxes += [detections[0]]
        #     detections = detections[~invalid]
        # if keep_boxes:
        #     output[image_i] = torch.stack(keep_boxes)

    return output


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        # TP
        true_positives = np.zeros(pred_boxes.shape[0])

        # Get current batch labels
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def xywh2p1p2(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.
    y[..., 1] = x[..., 1] - x[..., 3] / 2.
    y[..., 2] = x[..., 0] + x[..., 2] / 2.
    y[..., 3] = x[..., 1] + x[..., 3] / 2.
    return y


def bbox_wh_iou(wh1, wh2):
    """

    :param wh1: of shape (n, 2)
    :param wh2: of shape (m, 2)
    :return: (n, m)
    """
    wh1 = wh1.unsqueeze(-2)  # (n, 1, 2)
    wh2 = wh2.unsqueeze(0)  # (1, m, 2)

    w1, h1 = wh1[..., 0], wh1[..., 1]
    w2, h2 = wh2[..., 0], wh2[..., 1]

    inter_area = torch.min(w1, w2) * torch.min(h1, h2)  # (n, m)
    union_area = (w1 * h1 + epsilon) + w2 * h2 - inter_area  # (n, m)

    return inter_area / union_area


def bbox_iou(box1, box2, p1p2=True):
    """
    Returns the IoU of two bounding boxes
    :param box1: of shape (n, 4)
    :param box2: of shape (n, 4)
    :param p1p2: indicate the type of bbox is two points?
    :return: iou of shape (n,)
    """
    if not p1p2:
        box1_mins = box1[..., :2] - box1[..., 2:4] / 2.
        box1_maxes = box1[..., :2] + box1[..., 2:4] / 2.
        box2_mins = box2[..., :2] - box2[..., 2:4] / 2.
        box2_maxes = box2[..., :2] + box2[..., 2:4] / 2.
    else:
        box1_mins = box1[..., :2]
        box1_maxes = box1[..., 2:4]
        box2_mins = box2[..., :2]
        box2_maxes = box2[..., 2:4]

    inter_mins = torch.max(box1_mins, box2_mins)
    inter_maxes = torch.min(box1_maxes, box2_maxes)
    inter_wh = torch.clamp(inter_maxes - inter_mins + 1, min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    box1_area = (box1_maxes[..., 0] - box1_mins[..., 0] + 1) * (box1_maxes[..., 1] - box1_mins[..., 1] + 1)
    box2_area = (box2_maxes[..., 0] - box2_mins[..., 0] + 1) * (box2_maxes[..., 1] - box2_mins[..., 1] + 1)

    iou = inter_area / (box1_area + box2_area - inter_area + epsilon)
    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_threshold):
    """

    :param pred_boxes:  (batch, num_anchors, width, height, 4)
    :param pred_cls:    (batch, num_anchors, width, height, num_classes)
    :param target:      (num_ground_true_bboxes, 6)
    :param anchors:     (num_anchors, 2)
    :param ignore_threshold:
    :return:
    """

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    batch_size = pred_boxes.size(0)
    num_anchors = pred_boxes.size(1)
    num_classes = pred_cls.size(-1)
    grid_size = pred_boxes.size(2)

    # Indicate which anchor captured obj.
    obj_mask = ByteTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)

    # Indicate anchor that not captured obj.
    noobj_mask = ByteTensor(batch_size, num_anchors, grid_size, grid_size).fill_(1)

    class_mask = FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)
    iou_scores = FloatTensor(batch_size, num_anchors, grid_size, grid_size).fill_(0)

    tboxes = FloatTensor(batch_size, num_anchors, grid_size, grid_size, 4).fill_(0)
    tcls = FloatTensor(batch_size, num_anchors, grid_size, grid_size, num_classes).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * grid_size  # (n, 4)
    gxy = target_boxes[:, :2]  # (n, 2)
    gwh = target_boxes[:, 2:]  # (n, 2)

    # Find anchors with best iou with targets from all anchors
    ious = bbox_wh_iou(anchors, gwh)  # (num_anchors, n)
    best_ious, best_idx = ious.max(0)  # (n, ), (n, )

    # Separate target values
    b, target_labels = target[:, :2].long().t()  # (n, ), (n, )
    gi, gj = gxy.long().t()

    # Sets masks
    obj_mask[b, best_idx, gj, gi] = 1
    noobj_mask[b, best_idx, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    # for i, anchor_ious in enumerate(ious.t()):
    #     noobj_mask[b[i], anchor_ious > ignore_threshold, gj[i], gi[i]] = 0
    noobj_mask[b, :, gj, gi] = noobj_mask[b, :, gj, gi] * (ious.t() <= ignore_threshold)

    tboxes[b, best_idx, gj, gi, 0:2] = gxy - gxy.floor()
    tboxes[b, best_idx, gj, gi, 2:4] = torch.log(gwh / anchors[best_idx] + epsilon)

    # One-hot encoding of label
    tcls[b, best_idx, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_idx, gj, gi] = (pred_cls[b, best_idx, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_idx, gj, gi] = bbox_iou(pred_boxes[b, best_idx, gj, gi], target_boxes, p1p2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tboxes, tcls, tconf
