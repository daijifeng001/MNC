# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import cv2
import numpy as np
import yaml

import caffe
from mnc_config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


class MNCDataLayer(caffe.Layer):
    """
    Provide image, image w/h/scale, gt boxes/masks and mask info to upper layers
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._name_to_top_map = {}
        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = 0
        assert(cfg.TRAIN.HAS_RPN, 'Use RPN for this project')
        # Just pseudo setup
        top[1].reshape(1, 3)
        self._name_to_top_map['im_info'] = 1
        top[2].reshape(1, 4)
        self._name_to_top_map['gt_boxes'] = 2
        if cfg.MNC_MODE:
            top[3].reshape(1, 21, 21)
            self._name_to_top_map['gt_masks'] = 3
            top[4].reshape(1, 3)
            self._name_to_top_map['mask_info'] = 4
        assert len(top) == len(self._name_to_top_map)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*blob.shape)
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def set_maskdb(self, maskdb):
        self._maskdb = maskdb
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_image_blob(self, roidb, scale_inds):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = 1  # len(roidb)
        processed_ims = []
        im_scales = []
        for i in xrange(num_images):
            im = cv2.imread(roidb['image'])
            if roidb['flipped']:
                im = im[:, ::-1, :]
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                            cfg.TRAIN.MAX_SIZE)
            im_scales.append(im_scale)
            processed_ims.append(im)
        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)
        return blob, im_scales

    def _get_next_minibatch(self):
        """
        Return the blobs to be used for the next minibatch.
        """
        assert cfg.TRAIN.IMS_PER_BATCH == 1, 'Only single batch forwarding is supported'

        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur]
        self._cur += 1
        roidb = self._roidb[db_inds]
        
        random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=1)
        im_blob, im_scales = self._get_image_blob(roidb, random_scale_inds)

        gt_label = np.where(roidb['gt_classes'] != 0)[0]
        gt_boxes = np.hstack((roidb['boxes'][gt_label, :] * im_scales[0],
                              roidb['gt_classes'][gt_label, np.newaxis])).astype(np.float32)
        blobs = {
            'data': im_blob,
            'gt_boxes': gt_boxes,
            'im_info': np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
        }

        if cfg.MNC_MODE:
            maskdb = self._maskdb[db_inds]
            mask_list = maskdb['gt_masks']
            mask_max_x = maskdb['mask_max'][0]
            mask_max_y = maskdb['mask_max'][1]
            gt_masks = np.zeros((len(mask_list), mask_max_y, mask_max_x))
            mask_info = np.zeros((len(mask_list), 2))
            for j in xrange(len(mask_list)):
                mask = mask_list[j]
                mask_x = mask.shape[1]
                mask_y = mask.shape[0]
                gt_masks[j, 0:mask_y, 0:mask_x] = mask
                mask_info[j, 0] = mask_y
                mask_info[j, 1] = mask_x
            blobs['gt_masks'] = gt_masks
            blobs['mask_info'] = mask_info

        return blobs
