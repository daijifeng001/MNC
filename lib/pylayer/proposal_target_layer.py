# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from mnc_config import cfg
from transform.bbox_transform import \
    bbox_transform, bbox_compute_targets, \
    scale_boxes, get_bbox_regression_label
from transform.anchors import generate_anchors
from transform.mask_transform import intersect_mask
from utils.cython_bbox import bbox_overlaps


class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._anchors = generate_anchors()
        self._num_anchors = self._anchors.shape[0]
        self._num_classes = layer_params['num_classes']
        self._bp_all = layer_params.get('bp_all', True)
        self._top_name_map = {}
        top[0].reshape(1, 5)
        self._top_name_map['rois'] = 0
        top[1].reshape(1, 1)
        self._top_name_map['labels'] = 1
        top[2].reshape(1, self._num_classes * 4)
        self._top_name_map['bbox_targets'] = 2
        top[3].reshape(1, self._num_classes * 4)
        self._top_name_map['bbox_inside_weights'] = 3
        top[4].reshape(1, self._num_classes * 4)
        self._top_name_map['bbox_outside_weights'] = 4
        # Add mask-related information
        if cfg.MNC_MODE:
            top[5].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
            self._top_name_map['mask_targets'] = 5
            top[6].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
            self._top_name_map['mask_weight'] = 6
            top[7].reshape(1, 4)
            self._top_name_map['gt_masks_info'] = 7
            if cfg.TRAIN.MIX_INDEX:
                top[8].reshape(1, 4)
                self._top_name_map['fg_inds'] = 8
                top[9].reshape(1, 4)
                self._top_name_map['bg_inds'] = 9

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        im_info = bottom[2].data[0, :]
        im_scale = im_info[2]
        # get original masks
        if cfg.MNC_MODE:
            gt_masks = bottom[3].data
            mask_info = bottom[4].data
        else:
            gt_masks = None
            mask_info = None
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        # Sample rois with classification labels and bounding box regression targets

        blobs, fg_inds, bg_inds, keep_inds = _sample_rois(
            all_rois, gt_boxes, rois_per_image, self._num_classes, gt_masks, im_scale, mask_info)
        self._keep_ind = keep_inds if self._bp_all else fg_inds

        for blob_name, blob in blobs.iteritems():
            top[self._top_name_map[blob_name]].reshape(*blob.shape)
            top[self._top_name_map[blob_name]].data[...] = blob.astype(np.float32, copy=False)

        if cfg.TRAIN.MIX_INDEX:
            all_rois_index = bottom[5].data
            fg_inds = fg_inds[fg_inds < all_rois_index.shape[1]].astype(int)
            fg_inds = all_rois_index[0, fg_inds]
            bg_inds = all_rois_index[0, bg_inds.astype(int)]
            top[self._top_name_map['fg_inds']].reshape(*fg_inds.shape)
            top[self._top_name_map['fg_inds']].data[...] = fg_inds
            top[self._top_name_map['bg_inds']].reshape(*bg_inds.shape)
            top[self._top_name_map['bg_inds']].data[...] = bg_inds

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff.fill(0.)
            # Eliminate gt_inds from the keep inds
            valid_inds = np.where(self._keep_ind < bottom[0].diff.shape[0])[0]
            valid_bot_inds = self._keep_ind[valid_inds].astype(int)
            bottom[0].diff[valid_bot_inds, :] = top[0].diff[valid_inds, :]


def _sample_rois(all_rois, gt_boxes, rois_per_image, num_classes, gt_masks, im_scale, mask_info):
    """
    Generate a random sample of RoIs comprising
    foreground and background examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Sample foreground indexes
    fg_inds = []
    for i in xrange(len(cfg.TRAIN.FG_FRACTION)):
        cur_inds = np.where((max_overlaps >= cfg.TRAIN.FG_THRESH_LO[i]) &
                            (max_overlaps <= cfg.TRAIN.FG_THRESH_HI[i]))[0]
        cur_rois_this_image = min(cur_inds.size, np.round(rois_per_image *
                                                          cfg.TRAIN.FG_FRACTION[i]))
        if cur_inds.size > 0:
            cur_inds = npr.choice(cur_inds, size=cur_rois_this_image, replace=False)
        fg_inds = np.hstack((fg_inds, cur_inds))
        fg_inds = np.unique(fg_inds)
    fg_rois_per_image = fg_inds.size
    # Sample background indexes according to number of foreground
    bg_rois_per_this_image = rois_per_image - fg_rois_per_image
    bg_inds = []
    for i in xrange(len(cfg.TRAIN.BG_FRACTION)):
        cur_inds = np.where((max_overlaps >= cfg.TRAIN.BG_THRESH_LO[i]) &
                            (max_overlaps <= cfg.TRAIN.BG_THRESH_HI[i]))[0]
        cur_rois_this_image = min(cur_inds.size, np.round(bg_rois_per_this_image *
                                                          cfg.TRAIN.BG_FRACTION[i]))
        if cur_inds.size > 0:
            cur_inds = npr.choice(cur_inds, size=cur_rois_this_image, replace=False)
        bg_inds = np.hstack((bg_inds, cur_inds))
        bg_inds = np.unique(bg_inds)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds).astype(int)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = bbox_compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], normalize=True)
    bbox_target_data = np.hstack((labels[:, np.newaxis], bbox_target_data))\
        .astype(np.float32, copy=False)
    bbox_targets, bbox_inside_weights = get_bbox_regression_label(
        bbox_target_data, num_classes)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    blobs = {
        'rois': rois,
        'labels': labels,
        'bbox_targets': bbox_targets,
        'bbox_inside_weights': bbox_inside_weights,
        'bbox_outside_weights': bbox_outside_weights
    }

    if cfg.MNC_MODE:
        scaled_rois = rois[:, 1:5] / float(im_scale)

        # map to original image space
        scaled_gt_boxes = gt_boxes[:, :4] / float(im_scale)
        pos_masks = np.zeros((len(keep_inds), 1, cfg.MASK_SIZE,  cfg.MASK_SIZE))
        top_mask_info = np.zeros((len(keep_inds), 12))
        top_mask_info[len(fg_inds):, :] = -1

        for i, val in enumerate(fg_inds):
            gt_box = scaled_gt_boxes[gt_assignment[val]]
            gt_box = np.around(gt_box).astype(int)
            ex_box = np.around(scaled_rois[i]).astype(int)
            gt_mask = gt_masks[gt_assignment[val]]
            gt_mask_info = mask_info[gt_assignment[val]]
            gt_mask = gt_mask[0:gt_mask_info[0], 0:gt_mask_info[1]]
            # calculate mask regression targets
            # (intersection of bounding box and gt mask)
            ex_mask = intersect_mask(ex_box, gt_box, gt_mask)

            pos_masks[i, ...] = ex_mask
            top_mask_info[i, 0] = gt_assignment[val]
            top_mask_info[i, 1] = gt_mask_info[0]
            top_mask_info[i, 2] = gt_mask_info[1]
            top_mask_info[i, 3] = labels[i]

            top_mask_info[i, 4:8] = ex_box
            top_mask_info[i, 8:12] = gt_box

        mask_weight = np.zeros((rois.shape[0], 1, cfg.MASK_SIZE, cfg.MASK_SIZE))
        # only assign box-level foreground as positive mask regression
        mask_weight[0:len(fg_inds), :, :, :] = 1
        blobs['mask_targets'] = pos_masks
        blobs['mask_weight'] = mask_weight
        blobs['gt_masks_info'] = top_mask_info

    return blobs, fg_inds, bg_inds, keep_inds
