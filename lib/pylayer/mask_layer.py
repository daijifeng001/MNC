# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import caffe
import cv2
import numpy as np
from transform.mask_transform import mask_overlap
from mnc_config import cfg


class MaskLayer(caffe.Layer):
    """
    This layer Take input from sigmoid predicted masks
    Assign each label for segmentation classifier according
    to region overlap
    """

    def setup(self, bottom, top):
        self._phase = str(self.phase)
        self._top_name_map = {}
        top[0].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
        self._top_name_map['mask_proposal'] = 0
        if self._phase == 'TRAIN':
            top[1].reshape(1, 1)
            self._top_name_map['mask_proposal_label'] = 1

    def reshape(self, bottom, top):
        """
        Reshaping happens during the call to forward
        """
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
        if propagate_down[0]:
            bottom[0].diff.fill(0.)
            top_grad = top[0].diff.reshape(top[0].diff.shape[0], cfg.MASK_SIZE * cfg.MASK_SIZE)
            bottom[0].diff[self.pos_sample, :] = top_grad[self.pos_sample, :]

    def forward_train(self, bottom, top):
        # Take sigmoid prediction as input
        mask_pred = bottom[0].data
        # get ground truth mask and labels
        gt_masks = bottom[1].data
        gt_masks_info = bottom[2].data
        num_mask_pred = mask_pred.shape[0]
        top_label = np.zeros((gt_masks_info.shape[0], 1))
        # 2. Calculate region overlap
        #    Since the target gt mask may have different size
        #    We need to resize predicted masks into different sizes
        mask_size = cfg.MASK_SIZE
        for i in xrange(num_mask_pred):
            # if the bounding box is itself background
            if gt_masks_info[i][0] == -1:
                top_label[i][0] = 0
                continue
            else:
                info = gt_masks_info[i]
                gt_mask = gt_masks[info[0]][0:info[1], 0:info[2]]
                ex_mask = mask_pred[i].reshape((mask_size, mask_size))
                ex_box = np.round(info[4:8]).astype(int)
                gt_box = np.round(info[8:12]).astype(int)
                # resize to large gt_masks, note cv2.resize is column first
                ex_mask = cv2.resize(ex_mask.astype(np.float32), (ex_box[2] - ex_box[0] + 1,
                                                                  ex_box[3] - ex_box[1] + 1))
                ex_mask = ex_mask >= cfg.BINARIZE_THRESH
                top_label[i][0] = 0 if mask_overlap(ex_box, gt_box, ex_mask, gt_mask) < cfg.TRAIN.FG_SEG_THRESH else info[3]

        # output continuous mask for MNC
        resized_mask_pred = mask_pred.reshape((num_mask_pred, 1, cfg.MASK_SIZE, cfg.MASK_SIZE))
        self.pos_sample = np.where(top_label > 0)[0]

        blobs = {
            'mask_proposal': resized_mask_pred,
            'mask_proposal_label': top_label
        }
        return blobs

    def forward_test(self, bottom, top):
        mask_pred = bottom[0].data
        num_mask_pred = mask_pred.shape[0]
        resized_mask_pred = mask_pred.reshape((num_mask_pred, 1, cfg.MASK_SIZE, cfg.MASK_SIZE))
        blobs = {
            'mask_proposal': resized_mask_pred
        }
        return blobs
