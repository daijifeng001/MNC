# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
from utils.cython_bbox import bbox_overlaps
from mnc_config import cfg


def compute_targets(rois, overlaps, labels):
    """
    Compute bounding-box regression targets for an image.
    """
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap  each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets


def bbox_transform(ex_rois, gt_rois):
    """
    Compute bbox regression targets of external rois
    with respect to gt rois
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    """
    invert bounding box transform
    apply delta on anchors to get transformed proposals
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes inside image boundaries
    """
    x1 = boxes[:, 0::4]
    y1 = boxes[:, 1::4]
    x2 = boxes[:, 2::4]
    y2 = boxes[:, 3::4]
    keep = np.where((x1 >= 0) & (x2 <= im_shape[1] - 1) & (y1 >= 0) & (y2 <= im_shape[0] - 1))[0]
    clipped_boxes = np.zeros(boxes.shape, dtype=boxes.dtype)
    # x1 >= 0
    clipped_boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    clipped_boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    clipped_boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    clipped_boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return clipped_boxes, keep


def filter_small_boxes(boxes, min_size):
    """
    Remove all boxes with any side smaller than min_size.
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def scale_boxes(boxes, alpha):
    """
    Scale boxes from w/h to alpha * w/h while keep center unchanged
    Args:
        boxes: a set of boxes specified using x1, y1, x2, y2
        alpha: scaling factor

    Returns:
        boxes: boxes after applying scaling
    """
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    ctr_x = boxes[:, 0] + 0.5 * w
    ctr_y = boxes[:, 1] + 0.5 * h
    scaled_w = w * alpha
    scaled_h = h * alpha
    scaled_boxes = np.zeros(boxes.shape, dtype=boxes.dtype)
    scaled_boxes[:, 0] = ctr_x - 0.5 * scaled_w
    scaled_boxes[:, 1] = ctr_y - 0.5 * scaled_h
    scaled_boxes[:, 2] = ctr_x + 0.5 * scaled_w
    scaled_boxes[:, 3] = ctr_y + 0.5 * scaled_h
    return scaled_boxes


def bbox_compute_targets(ex_rois, gt_rois, normalize):
    """
    Compute bounding-box regression targets for an image
    Parameters:
    -----------
    ex_rois: ROIs from external source (anchors or proposals)
    gt_rois: ground truth ROIs
    normalize: whether normalize box (since RPN doesn't need to normalize)

    Returns:
    -----------
    Relative value for anchor or proposals
    """
    assert ex_rois.shape == gt_rois.shape

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED and normalize:
        # Optionally normalize targets by a precomputed mean and std
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) /
                   np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    return targets.astype(np.float32, copy=False)


def get_bbox_regression_label(bbox_target_data, num_class):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    assert bbox_target_data.shape[1] == 5
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_class), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
