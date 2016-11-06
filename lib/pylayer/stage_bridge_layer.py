# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from transform.bbox_transform import \
    bbox_transform_inv, bbox_compute_targets, \
    clip_boxes, get_bbox_regression_label
from transform.mask_transform import intersect_mask
from mnc_config import cfg
from utils.cython_bbox import bbox_overlaps


class StageBridgeLayer(caffe.Layer):
    """
    This layer take input from bounding box prediction
    and output a set of new rois after applying transformation
    It will also provide mask/bbox regression targets
    during training phase
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        # bottom 0 is ~ n ROIs to train Fast RCNN
        # bottom 1 is ~ n * 4(1+c) bbox prediction
        # bottom 2 is ~ n * (1+c) bbox scores (seg classification)
        self._phase = str(self.phase)
        if self._phase == 'TRAIN':
            self._use_clip = layer_params['use_clip']
            self._clip_denominator = float(layer_params.get('clip_base', 64))
            self._clip_thresh = 1.0 / self._clip_denominator
            self._feat_stride = layer_params['feat_stride']
            self._num_classes = layer_params['num_classes']

        # meaning of top blobs speak for themselves
        self._top_name_map = {}
        if self._phase == 'TRAIN':
            top[0].reshape(1, 5)
            self._top_name_map['rois'] = 0
            top[1].reshape(1, 1)
            self._top_name_map['labels'] = 1
            top[2].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
            self._top_name_map['mask_targets'] = 2
            top[3].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
            self._top_name_map['mask_weight'] = 3
            top[4].reshape(1, 4)
            self._top_name_map['gt_mask_info'] = 4
            top[5].reshape(1, 21 * 4)
            self._top_name_map['bbox_targets'] = 5
            top[6].reshape(1, 21 * 4)
            self._top_name_map['bbox_inside_weights'] = 6
            top[7].reshape(1, 21 * 4)
            self._top_name_map['bbox_outside_weights'] = 7
        elif self._phase == 'TEST':
            top[0].reshape(1, 5)
            self._top_name_map['rois'] = 0
        else:
            print 'Unrecognized phase'
            raise NotImplementedError

    def reshape(self, bottom, top):
        # reshape happens during forward
        pass

    def forward(self, bottom, top):
        if str(self.phase) == 'TRAIN':
            blobs = self.forward_train(bottom, top)
        elif str(self.phase) == 'TEST':
            blobs = self.forward_test(bottom, top)
        else:
            print 'Unrecognized phase'
            raise NotImplementedError

        for blob_name, blob in blobs.iteritems():
            top[self._top_name_map[blob_name]].reshape(*blob.shape)
            top[self._top_name_map[blob_name]].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """
        Description:
            We need to implement bp for 2 bottoms:
            The top diff is x_new, y_new, w_new, h_new
        """
        deltas = bottom[1].data
        dfdxc = top[0].diff[:, 1]
        dfdyc = top[0].diff[:, 2]
        dfdw = top[0].diff[:, 3]
        dfdh = top[0].diff[:, 4]
        W_old = bottom[0].data[:, 2] - bottom[0].data[:, 0]
        H_old = bottom[0].data[:, 3] - bottom[0].data[:, 1]

        if propagate_down[0]:
            bottom[0].diff.fill(0.)
            for ind, i in enumerate(self._keep_inds):
                if i >= bottom[0].diff.shape[0] or self._bbox_reg_labels[i] == 0:
                    continue
                delta_x = deltas[i, 4*self._bbox_reg_labels[i]]
                delta_y = deltas[i, 4*self._bbox_reg_labels[i]+1]
                delta_w = deltas[i, 4*self._bbox_reg_labels[i]+2]
                delta_h = deltas[i, 4*self._bbox_reg_labels[i]+3]
                bottom[0].diff[i, 1] = dfdxc[ind]
                bottom[0].diff[i, 2] = dfdyc[ind]
                bottom[0].diff[i, 3] = dfdw[ind] * (delta_x + np.exp(delta_w))
                bottom[0].diff[i, 4] = dfdh[ind] * (delta_y + np.exp(delta_h))

        if propagate_down[1]:
            bottom[1].diff.fill(0.)
            for ind, i in enumerate(self._keep_inds):
                if i >= bottom[1].diff.shape[0] or i not in self._clip_keep or self._bbox_reg_labels[i] == 0:
                    continue
                delta_w = deltas[i, 4*self._bbox_reg_labels[i]+2]
                delta_h = deltas[i, 4*self._bbox_reg_labels[i]+3]
                bottom[1].diff[i, 4*self._bbox_reg_labels[i]] = dfdxc[ind] * W_old[i]
                bottom[1].diff[i, 4*self._bbox_reg_labels[i]+1] = dfdyc[ind] * H_old[i]
                bottom[1].diff[i, 4*self._bbox_reg_labels[i]+2] = dfdw[ind] * np.exp(delta_w) * W_old[i]
                bottom[1].diff[i, 4*self._bbox_reg_labels[i]+3] = dfdh[ind] * np.exp(delta_h) * H_old[i]
                if self._use_clip:
                    bottom[1].diff[i, 4*self._bbox_reg_labels[i]] = np.minimum(np.maximum(
                        bottom[1].diff[i, 4*self._bbox_reg_labels[i]], -self._clip_thresh), self._clip_thresh)
                    bottom[1].diff[i, 4*self._bbox_reg_labels[i]+1] = np.minimum(np.maximum(
                        bottom[1].diff[i, 4*self._bbox_reg_labels[i]+1], -self._clip_thresh), self._clip_thresh)
                    bottom[1].diff[i, 4*self._bbox_reg_labels[i]+2] = np.minimum(np.maximum(
                        bottom[1].diff[i, 4*self._bbox_reg_labels[i]+2], -self._clip_thresh), self._clip_thresh)
                    bottom[1].diff[i, 4*self._bbox_reg_labels[i]+3] = np.minimum(np.maximum(
                        bottom[1].diff[i, 4*self._bbox_reg_labels[i]+3], -self._clip_thresh), self._clip_thresh)

    def forward_train(self, bottom, top):
        """
        During forward, we need to do several things:
        1. Apply bounding box regression output which has highest
           classification score to proposed ROIs
        2. Sample ROIs based on there current overlaps, assign labels
           on them
        3. Make mask regression targets and positive/negative weights,
           just like the proposal_target_layer
        """
        rois = bottom[0].data
        bbox_deltas = bottom[1].data
        # Apply bounding box regression according to maximum segmentation score
        seg_scores = bottom[2].data
        self._bbox_reg_labels = seg_scores[:, 1:].argmax(axis=1) + 1

        gt_boxes = bottom[3].data
        gt_masks = bottom[4].data
        im_info = bottom[5].data[0, :]
        mask_info = bottom[6].data

        # select bbox_deltas according to
        artificial_deltas = np.zeros((rois.shape[0], 4))
        for i in xrange(rois.shape[0]):
            artificial_deltas[i, :] = bbox_deltas[i, 4*self._bbox_reg_labels[i]:4*(self._bbox_reg_labels[i]+1)]
        artificial_deltas[self._bbox_reg_labels == 0, :] = 0

        all_rois = np.zeros((rois.shape[0], 5))
        all_rois[:, 0] = 0
        all_rois[:, 1:5] = bbox_transform_inv(rois[:, 1:5], artificial_deltas)
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        all_rois[:, 1:5], self._clip_keep = clip_boxes(all_rois[:, 1:5], im_info[:2])

        labels, rois_out, fg_inds, keep_inds, mask_targets, top_mask_info, bbox_targets, bbox_inside_weights = \
            self._sample_output(all_rois, gt_boxes, im_info[2], gt_masks, mask_info)
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
        self._keep_inds = keep_inds

        mask_weight = np.zeros((rois_out.shape[0], 1, cfg.MASK_SIZE, cfg.MASK_SIZE))
        mask_weight[0:len(fg_inds), :, :, :] = 1

        blobs = {
            'rois': rois_out,
            'labels': labels,
            'mask_targets': mask_targets,
            'mask_weight': mask_weight,
            'gt_mask_info': top_mask_info,
            'bbox_targets': bbox_targets,
            'bbox_inside_weights': bbox_inside_weights,
            'bbox_outside_weights': bbox_outside_weights
        }
        return blobs

    def _sample_output(self, all_rois, gt_boxes, im_scale, gt_masks, mask_info):
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]
        # Sample foreground indexes
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.BBOX_THRESH)[0]
        bg_inds = np.where(max_overlaps < cfg.TRAIN.BBOX_THRESH)[0]
        keep_inds = np.append(fg_inds, bg_inds).astype(int)
        # Select sampled values from various arrays:
        labels = labels[keep_inds]
        # Clamp labels for the background RoIs to 0
        labels[len(fg_inds):] = 0
        rois = all_rois[keep_inds]

        bbox_target_data = bbox_compute_targets(
            rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], normalize=True)
        bbox_target_data = np.hstack((labels[:, np.newaxis], bbox_target_data))\
            .astype(np.float32, copy=False)
        bbox_targets, bbox_inside_weights = get_bbox_regression_label(
            bbox_target_data, self._num_classes)

        scaled_rois = rois[:, 1:5] / float(im_scale)
        scaled_gt_boxes = gt_boxes[:, :4] / float(im_scale)

        pos_masks = np.zeros((len(keep_inds), 1,  cfg.MASK_SIZE,  cfg.MASK_SIZE))
        top_mask_info = np.zeros((len(keep_inds), 12))
        top_mask_info[len(fg_inds):, :] = -1

        for i, val in enumerate(fg_inds):
            gt_box = scaled_gt_boxes[gt_assignment[val]]
            gt_box = np.around(gt_box).astype(int)
            ex_box = np.around(scaled_rois[i]).astype(int)
            gt_mask = gt_masks[gt_assignment[val]]
            gt_mask_info = mask_info[gt_assignment[val]]
            gt_mask = gt_mask[0:gt_mask_info[0], 0:gt_mask_info[1]]
            # regression targets is the intersection of bounding box and gt mask
            ex_mask = intersect_mask(ex_box, gt_box, gt_mask)
            pos_masks[i, ...] = ex_mask
            top_mask_info[i, 0] = gt_assignment[val]
            top_mask_info[i, 1] = gt_mask_info[0]
            top_mask_info[i, 2] = gt_mask_info[1]
            top_mask_info[i, 3] = labels[i]
            top_mask_info[i, 4:8] = ex_box
            top_mask_info[i, 8:12] = gt_box

        return labels, rois, fg_inds, keep_inds, pos_masks, top_mask_info, bbox_targets, bbox_inside_weights

    def forward_test(self, bottom, top):
        rois = bottom[0].data
        bbox_deltas = bottom[1].data
        # get ~ n * 4(1+c) new rois
        all_rois = bbox_transform_inv(rois[:, 1:5], bbox_deltas)
        scores = bottom[2].data
        im_info = bottom[3].data
        # get highest scored category's bounding box regressor
        score_max = scores.argmax(axis=1)
        rois_out = np.zeros((rois.shape[0], 5))
        # Single batch training
        rois_out[:, 0] = 0
        for i in xrange(len(score_max)):
            rois_out[i, 1:5] = all_rois[i, 4*score_max[i]:4*(score_max[i]+1)]
        rois_out[:, 1:5], _ = clip_boxes(rois_out[:, 1:5], im_info[0, :2])
        blobs = {
            'rois': rois_out
        }
        return blobs
