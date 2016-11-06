# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import caffe
import numpy as np
import yaml

from mnc_config import cfg
from transform.anchors import generate_anchors
from transform.bbox_transform import clip_boxes, bbox_transform_inv, filter_small_boxes
from nms.nms_wrapper import nms

DEBUG = False
PRINT_GRADIENT = 1


class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._feat_stride = layer_params['feat_stride']
        self._anchors = generate_anchors()
        self._num_anchors = self._anchors.shape[0]
        self._use_clip = layer_params.get('use_clip', 0)
        self._clip_denominator = float(layer_params.get('clip_base', 256))
        self._clip_thresh = 1.0 / self._clip_denominator
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        self._top_name_map = {}
        top[0].reshape(1, 5)
        self._top_name_map['rois'] = 0
        # For MNC, we force the output proposals will also be used to train RPN
        # this is achieved by passing proposal_index to anchor_target_layer
        if str(self.phase) == 'TRAIN':
            if cfg.TRAIN.MIX_INDEX:
                top[1].reshape(1, 1)
                self._top_name_map['proposal_index'] = 1

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted transform deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        assert bottom[0].data.shape[0] == 1, 'Only single item batches are supported'

        cfg_key = str(self.phase)  # either 'TRAIN' or 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        # 1. Generate proposals from transform deltas and shifted anchors
        height, width = scores.shape[-2:]
        self._height = height
        self._width = width
        # Enumerate all shifts
        shift_x = np.arange(0, self._width) * self._feat_stride
        shift_y = np.arange(0, self._height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        _, keep = clip_boxes(anchors, im_info[:2])
        self._anchor_index_before_clip = keep

        # Transpose and reshape predicted transform transformations to get them
        # into the same order as the anchors:
        #
        # transform deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via transform transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals, keep = clip_boxes(proposals, im_info[:2])
        # Record the cooresponding index before and after clip
        # This step doesn't need unmap
        # We need it to decide whether do back propagation
        self._proposal_index_before_clip = keep

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = filter_small_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]
        self._ind_after_filter = keep

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]

        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
        self._ind_after_sort = order
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)

        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]

        scores = scores[keep]
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        proposals = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        self._proposal_index = keep

        blobs = {
            'rois': proposals
        }

        if str(self.phase) == 'TRAIN':
            if cfg.TRAIN.MIX_INDEX:
                all_rois_index = self._ind_after_filter[self._ind_after_sort[self._proposal_index]].reshape(1, len(keep))
                blobs['proposal_index'] = all_rois_index

        # Copy data to forward to top layer
        for blob_name, blob in blobs.iteritems():
            top[self._top_name_map[blob_name]].reshape(*blob.shape)
            top[self._top_name_map[blob_name]].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):

        if propagate_down[1]:
            bottom[1].diff.fill(0.0)

            # first count only non-zero top gradient to accelerate computing
            top_non_zero_ind = np.unique(np.where(abs(top[0].diff[:, :]) > 0)[0])
            proposal_index = np.asarray(self._proposal_index)
            # unmap indexes to the original scale
            unmap_val = self._ind_after_filter[self._ind_after_sort[proposal_index[top_non_zero_ind]]]

            # not back propagate gradient if proposals/anchors are out of image boundary
            # this is a 0/1 mask so we just multiply them when calculating bottom gradient
            weight_out_proposal = np.in1d(unmap_val, self._proposal_index_before_clip)
            weight_out_anchor = np.in1d(unmap_val, self._anchor_index_before_clip)

            # unmap_val are arranged as (H * W * A) as stated in forward comment
            # with A as the fastest dimension (which is different from caffe)
            c = unmap_val % self._num_anchors
            w = (unmap_val / self._num_anchors) % self._width
            h = (unmap_val / self._num_anchors / self._width) % self._height

            # width and height should be in feature map scale
            anchor_w = (self._anchors[c, 2] - self._anchors[c, 0])
            anchor_h = (self._anchors[c, 3] - self._anchors[c, 1])
            dfdx1 = top[0].diff[top_non_zero_ind, 1]
            dfdy1 = top[0].diff[top_non_zero_ind, 2]
            dfdx2 = top[0].diff[top_non_zero_ind, 3]
            dfdy2 = top[0].diff[top_non_zero_ind, 4]

            dfdxc = dfdx1 + dfdx2
            dfdyc = dfdy1 + dfdy2
            dfdw = 0.5 * (dfdx2 - dfdx1)
            dfdh = 0.5 * (dfdy2 - dfdy1)

            bottom[1].diff[0, 4*c, h, w] = \
                dfdxc * anchor_w * weight_out_proposal * weight_out_anchor
            bottom[1].diff[0, 4*c+1, h, w] = \
                dfdyc * anchor_h * weight_out_proposal * weight_out_anchor
            bottom[1].diff[0, 4*c+2, h, w] = \
                dfdw * np.exp(bottom[1].data[0, 4*c+2, h, w]) * anchor_w * weight_out_proposal * weight_out_anchor
            bottom[1].diff[0, 4*c+3, h, w] = \
                dfdh * np.exp(bottom[1].data[0, 4*c+3, h, w]) * anchor_h * weight_out_proposal * weight_out_anchor

            # if use gradient clip, constraint gradient inside [-thresh, thresh]
            if self._use_clip:
                bottom[1].diff[0, 4*c, h, w] = np.minimum(np.maximum(
                    bottom[1].diff[0, 4*c, h, w], -self._clip_thresh), self._clip_thresh)
                bottom[1].diff[0, 4*c+1, h, w] = np.minimum(np.maximum(
                    bottom[1].diff[0, 4*c+1, h, w], -self._clip_thresh), self._clip_thresh)
                bottom[1].diff[0, 4*c+2, h, w] = np.minimum(np.maximum(
                    bottom[1].diff[0, 4*c+2, h, w], -self._clip_thresh), self._clip_thresh)
                bottom[1].diff[0, 4*c+3, h, w] = np.minimum(np.maximum(
                    bottom[1].diff[0, 4*c+3, h, w], -self._clip_thresh), self._clip_thresh)
