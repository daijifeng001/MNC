# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import cv2
from mnc_config import cfg
from nms.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps
from nms.mv import mv


def mask_overlap(box1, box2, mask1, mask2):
    """
    This function calculate region IOU when masks are
    inside different boxes
    Returns:
        intersection over unions of this two masks
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 > x2 or y1 > y2:
        return 0
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # get masks in the intersection part
    start_ya = y1 - box1[1]
    start_xa = x1 - box1[0]
    inter_maska = mask1[start_ya: start_ya + h, start_xa:start_xa + w]

    start_yb = y1 - box2[1]
    start_xb = x1 - box2[0]
    inter_maskb = mask2[start_yb: start_yb + h, start_xb:start_xb + w]

    assert inter_maska.shape == inter_maskb.shape

    inter = np.logical_and(inter_maskb, inter_maska).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)


def intersect_mask(ex_box, gt_box, gt_mask):
    """
    This function calculate the intersection part of a external box
    and gt_box, mask it according to gt_mask

    Args:
        ex_box: external ROIS
        gt_box: ground truth boxes
        gt_mask: ground truth masks, not been resized yet
    Returns:
        regression_target: logical numpy array
    """
    x1 = max(ex_box[0], gt_box[0])
    y1 = max(ex_box[1], gt_box[1])
    x2 = min(ex_box[2], gt_box[2])
    y2 = min(ex_box[3], gt_box[3])
    if x1 > x2 or y1 > y2:
        return np.zeros((21, 21), dtype=bool)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    ex_starty = y1 - ex_box[1]
    ex_startx = x1 - ex_box[0]

    gt_starty = y1 - gt_box[1]
    gt_startx = x1 - gt_box[0]
    inter_maskb = gt_mask[gt_starty: gt_starty + h, gt_startx: gt_startx + w]
    regression_target = np.zeros((ex_box[3] - ex_box[1] + 1, ex_box[2] - ex_box[0] + 1))
    regression_target[ex_starty: ex_starty + h, ex_startx: ex_startx + w] = inter_maskb
    regression_target = regression_target.astype(np.float32)
    regression_target = cv2.resize(regression_target, (cfg.MASK_SIZE, cfg.MASK_SIZE))
    regression_target = regression_target >= cfg.BINARIZE_THRESH
    return regression_target


def clip_masked_boxes(boxes, masks, im_shape):
    """
    Clipped masked boxes inside image boundary
    """
    num_box = boxes.shape[0]
    for i in xrange(num_box):
        box = np.round(boxes[i]).astype(int)
        mask = cv2.resize(masks[i, 0].astype(np.float32), (box[2] - box[0] + 1, box[3] - box[1] + 1))
        clip_x1 = max(0, 0 - box[0])
        clip_y1 = max(0, 0 - box[1])
        clip_width = min(box[2], im_shape[1] - 1) - clip_x1
        clip_height = min(box[3], im_shape[0] - 1) - clip_y1
        clip_x2 = clip_x1 + clip_width
        clip_y2 = clip_y1 + clip_height
        mask = mask[clip_y1:clip_y2, clip_x1:clip_x2]
        masks[i, 0] = cv2.resize(mask.astype(np.float32), (cfg.MASK_SIZE, cfg.MASK_SIZE))
        box[0] = clip_x1
        box[1] = clip_y1
        box[2] = clip_x2
        box[3] = clip_y2
        boxes[i] = box
    return boxes, masks


def mask_aggregation(boxes, masks, mask_weights, im_width, im_height):
    """
    This function implements mask voting mechanism to give finer mask
    n is the candidate boxes (masks) number
    Args:
        masks: All masks need to be aggregated (n x sz x sz)
        mask_weights: class score associated with each mask (n x 1)
        boxes: tight box enclose each mask (n x 4)
        im_width, im_height: image information
    TODO: Ensure mask size is sz x sz or tight box size
    """
    assert boxes.shape[0] == len(masks) and boxes.shape[0] == mask_weights.shape[0]
    im_mask = np.zeros((im_height, im_width))
    for mask_ind in xrange(len(masks)):
        box = np.round(boxes[mask_ind])
        mask = (masks[mask_ind] >= cfg.BINARIZE_THRESH).astype(float)
        mask_weight = mask_weights[mask_ind]
        im_mask[box[1]:box[3]+1, box[0]:box[2]+1] += mask * mask_weight
    [r, c] = np.where(im_mask >= cfg.BINARIZE_THRESH)
    if len(r) == 0 or len(c) == 0:
        min_y = np.ceil(im_height / 2)
        min_x = np.ceil(im_width / 2)
        max_y = min_y
        max_x = min_x
    else:
        min_y = np.min(r)
        min_x = np.min(c)
        max_y = np.max(r)
        max_x = np.max(c)

    clipped_mask = im_mask[min_y:max_y+1, min_x:max_x+1]
    clipped_box = np.array((min_x, min_y, max_x, max_y), dtype=np.float32)
    return clipped_mask, clipped_box


def cpu_mask_voting(masks, boxes, scores, num_classes, max_per_image, im_width, im_height):
    """
    Wrapper function for mask voting, note we already know the class of boxes and masks
    Args:
        masks: ~ n x mask_sz x mask_sz
        boxes: ~ n x 4
        scores: ~ n x 1
        max_per_image: default would be 100
        im_width: width of image
        im_height: height of image
    """
    # apply nms and sort to get first images according to their scores
    scores = scores[:, 1:]
    num_detect = boxes.shape[0]
    res_mask = [[] for _ in xrange(num_detect)]
    for i in xrange(num_detect):
        box = np.round(boxes[i]).astype(int)
        mask = cv2.resize(masks[i, 0].astype(np.float32), (box[2] - box[0] + 1, box[3] - box[1] + 1))
        res_mask[i] = mask
    # Intermediate results
    sup_boxes = []
    sup_masks = []
    sup_scores = []
    tobesort_scores = []

    for i in xrange(num_classes - 1):
        dets = np.hstack((boxes.astype(np.float32), scores[:, i:i+1]))
        inds = nms(dets, cfg.TEST.MASK_MERGE_NMS_THRESH)
        ind_boxes = boxes[inds]
        ind_masks = masks[inds]
        ind_scores = scores[inds, i]
        order = ind_scores.ravel().argsort()[::-1]
        num_keep = min(len(order), max_per_image)
        order = order[0:num_keep]
        sup_boxes.append(ind_boxes[order])
        sup_masks.append(ind_masks[order])
        sup_scores.append(ind_scores[order])
        tobesort_scores.extend(ind_scores[order])

    sorted_scores = np.sort(tobesort_scores)[::-1]
    num_keep = min(len(sorted_scores), max_per_image)
    thresh = sorted_scores[num_keep-1]
    result_box = []
    result_mask = []
    for c in xrange(num_classes - 1):
        cls_box = sup_boxes[c]
        cls_score = sup_scores[c]
        keep = np.where(cls_score >= thresh)[0]
        new_sup_boxes = cls_box[keep]
        num_sup_box = len(new_sup_boxes)
        masks_ar = np.zeros((num_sup_box, 1, cfg.MASK_SIZE, cfg.MASK_SIZE))
        boxes_ar = np.zeros((num_sup_box, 4))
        for i in xrange(num_sup_box):
            # Get weights according to their segmentation scores
            cur_ov = bbox_overlaps(boxes.astype(np.float), new_sup_boxes[i, np.newaxis].astype(np.float))
            cur_inds = np.where(cur_ov >= cfg.TEST.MASK_MERGE_IOU_THRESH)[0]
            cur_weights = scores[cur_inds, c]
            cur_weights = cur_weights / sum(cur_weights)
            # Re-format mask when passing it to mask_aggregation
            pass_mask = [res_mask[j] for j in list(cur_inds)]
            # do mask aggregation
            tmp_mask, boxes_ar[i] = mask_aggregation(boxes[cur_inds], pass_mask, cur_weights, im_width, im_height)
            tmp_mask = cv2.resize(tmp_mask.astype(np.float32), (cfg.MASK_SIZE, cfg.MASK_SIZE))
            masks_ar[i, 0] = tmp_mask
        # make new array such that scores is the last dimension of boxes
        boxes_scored_ar = np.hstack((boxes_ar, cls_score[keep, np.newaxis]))
        result_box.append(boxes_scored_ar)
        result_mask.append(masks_ar)
    return result_box, result_mask


def gpu_mask_voting(masks, boxes, scores, num_classes, max_per_image, im_width, im_height):
    """
    A wrapper function, note we already know the class of boxes and masks
    Args:
        masks: ~ 300 x 21 x 21
        boxes: ~ 300 x 4
        scores: ~ 300 x 1
        max_per_image: default would be 100
        im_width:
        im_height:
    """
    # Intermediate results
    sup_boxes = []
    sup_scores = []
    tobesort_scores = []
    for i in xrange(num_classes):
        if i == 0:
            sup_boxes.append([])
            sup_scores.append([])
            continue
        dets = np.hstack((boxes.astype(np.float32), scores[:, i:i+1]))
        inds = nms(dets, cfg.TEST.MASK_MERGE_NMS_THRESH)
        ind_boxes = boxes[inds]
        ind_scores = scores[inds, i]
        num_keep = min(len(ind_scores), max_per_image)
        sup_boxes.append(ind_boxes[0:num_keep, :])
        sup_scores.append(ind_scores[0:num_keep])
        tobesort_scores.extend(ind_scores[0:num_keep])

    sorted_scores = np.sort(tobesort_scores)[::-1]
    num_keep = min(len(sorted_scores), max_per_image)
    thresh = sorted_scores[num_keep-1]
    # inds array to record which mask should be aggregated together
    candidate_inds = []
    # weight for each element in the candidate inds
    candidate_weights = []
    # start position for candidate array
    candidate_start = []
    candidate_scores = []
    class_bar = []
    for c in xrange(num_classes):
        if c == 0:
            continue
        cls_box = sup_boxes[c]
        cls_score = sup_scores[c]
        keep = np.where(cls_score >= thresh)[0]
        new_sup_boxes = cls_box[keep]
        num_sup_box = len(new_sup_boxes)
        for i in xrange(num_sup_box):
            cur_ov = bbox_overlaps(boxes.astype(np.float), new_sup_boxes[i, np.newaxis].astype(np.float))
            cur_inds = np.where(cur_ov >= cfg.TEST.MASK_MERGE_IOU_THRESH)[0]
            candidate_inds.extend(cur_inds)
            cur_weights = scores[cur_inds, c]
            cur_weights = cur_weights / sum(cur_weights)
            candidate_weights.extend(cur_weights)
            candidate_start.append(len(candidate_inds))
        candidate_scores.extend(cls_score[keep])
        class_bar.append(len(candidate_scores))
    candidate_inds = np.array(candidate_inds, dtype=np.int32)
    candidate_weights = np.array(candidate_weights, dtype=np.float32)
    candidate_start = np.array(candidate_start, dtype=np.int32)
    candidate_scores = np.array(candidate_scores, dtype=np.float32)
    result_mask, result_box = mv(boxes.astype(np.float32), masks, candidate_inds, candidate_start, candidate_weights, im_height, im_width)
    result_box = np.hstack((result_box, candidate_scores[:, np.newaxis]))
    list_result_box = []
    list_result_mask = []
    # separate result mask into different classes
    for i in xrange(num_classes - 1):
        cls_start = class_bar[i - 1] if i > 0 else 0
        cls_end = class_bar[i]
        list_result_box.append(result_box[cls_start:cls_end, :])
        list_result_mask.append(result_mask[cls_start:cls_end, :, :, :])

    return list_result_mask, list_result_box
