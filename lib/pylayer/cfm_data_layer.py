# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import cv2
import yaml
import scipy
import numpy as np
import numpy.random as npr
import caffe
from mnc_config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from transform.bbox_transform import get_bbox_regression_label, bbox_compute_targets


class CFMDataLayer(caffe.Layer):
    """
    Provide image, image w/h/scale, gt boxes/masks and mask info to upper layers
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._name_to_top_map = {}
        self.input_mz = cfg.TEST.CFM_INPUT_MASK_SIZE
        # For CFM architect, we have nine entries since there is no intermediate layer
        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = 0

        top[1].reshape(1, 4)
        self._name_to_top_map['rois'] = 1

        top[2].reshape(1, 1, self.input_mz, self.input_mz)
        self._name_to_top_map['masks'] = 2

        top[3].reshape(1, 1)
        self._name_to_top_map['box_label'] = 3

        top[4].reshape(1, 1)
        self._name_to_top_map['mask_label'] = 4

        top[5].reshape(1, self._num_classes * 4)
        self._name_to_top_map['bbox_targets'] = 5

        top[6].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
        self._name_to_top_map['mask_targets'] = 6

        top[7].reshape(1, self._num_classes * 4)
        self._name_to_top_map['bbox_inside_weights'] = 7

        top[8].reshape(1, self._num_classes * 4)
        self._name_to_top_map['bbox_outside_weights'] = 8

        top[9].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
        self._name_to_top_map['mask_weight'] = 9
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

    def set_image_info(self, imdb, mean, std):
        self.imdb = imdb
        self._mean = mean
        self._std = std
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            import PIL
            num_images = len(self.imdb.image_index)
            width_r = [PIL.Image.open(self.imdb.image_path_at(i)).size[0] for i in xrange(num_images)]
            height_r = [PIL.Image.open(self.imdb.image_path_at(i)).size[0] for i in xrange(num_images)]
            widths = np.array([width_r[i] for i in xrange(len(width_r))])
            heights = np.array([height_r[i] for i in xrange(len(height_r))])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(np.hstack((inds, inds+num_images)), (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_image_blob(self, roidb, scale_inds, im_names):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        processed_ims = []
        im_scales = []
        for i in xrange(num_images):
            im = cv2.imread(im_names[i])
            # here [0][0] is due to the nature of scipy.io.savemat
            # since it will change True/False to [[1]] or [[0]] with shape (1,1)
            # so we judge whether flip image in this un-normal way
            if roidb[i]['Flip'][0][0]:
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
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._perm):
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        total_imgs = self.imdb.num_images

        roidbs = []
        img_names = []
        for db_ind in list(db_inds):
            cache_dir = self.imdb.roidb_path_at(db_ind)
            roidb = scipy.io.loadmat(cache_dir)
            roidbs.append(roidb)
            img_names.append(self.imdb.image_path_at(db_ind % total_imgs))

        blobs = self._sample_blobs(roidbs, img_names)
        return blobs

    def _sample_blobs(self, roidbs, img_names):

        random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=cfg.TRAIN.IMS_PER_BATCH)
        im_blob, im_scales = self._get_image_blob(roidbs, random_scale_inds, img_names)

        rois_per_img = cfg.TRAIN.BATCH_SIZE / cfg.TRAIN.IMS_PER_BATCH

        rois_blob = np.zeros((0, 5), dtype=np.float32)
        masks_blob = np.zeros((0, 1, self.input_mz, self.input_mz))
        box_labels_blob = np.zeros((0, 1))
        mask_labels_blob = np.zeros((0, 1))
        bbox_targets_blob = np.zeros((0, self._num_classes * 4))
        mask_targets_blob = np.zeros((0, 1, cfg.MASK_SIZE, cfg.MASK_SIZE))
        bbox_inside_weights_blob = np.zeros((0, self._num_classes * 4))
        bbox_outside_weights_blob = np.zeros((0, self._num_classes * 4))
        mask_weights_blob = np.zeros((0, 1, cfg.MASK_SIZE, cfg.MASK_SIZE))

        for im_i, roidb in enumerate(roidbs):
            # Sample positive/negative using box-level overlap
            det_overlap = roidb['det_overlap']
            num_gt = len(roidb['gt_classes'])
            fg_det_inds = np.where(det_overlap >= cfg.TRAIN.FG_DET_THRESH)
            keep_inds = []
            for i in xrange(len(cfg.TRAIN.FRACTION_SAMPLE)):
                cur_keep_inds = np.where((det_overlap >= cfg.TRAIN.THRESH_LO_SAMPLE[i]) &
                                         (det_overlap <= cfg.TRAIN.THRESH_HI_SAMPLE[i]))[0]
                cur_rois_this_image = np.round(rois_per_img * cfg.TRAIN.FRACTION_SAMPLE[i])
                cur_rois_this_image = min(cur_rois_this_image, len(cur_keep_inds))
                if cur_keep_inds.size > 0:
                    cur_keep_inds = npr.choice(cur_keep_inds, size=cur_rois_this_image, replace=False)

                if i == 0:
                    keep_inds = cur_keep_inds
                else:
                    keep_inds = np.unique(np.hstack((keep_inds, cur_keep_inds)))

            fg_inds_det = keep_inds[np.in1d(keep_inds, fg_det_inds)]
            bg_inds_det = keep_inds[np.in1d(keep_inds, fg_det_inds, invert=True)]
            keep_inds = np.append(fg_inds_det, bg_inds_det).astype(int)
            # Assign box-level label and mask-level label
            input_box_labels = roidb['output_label'][keep_inds]
            # input_box_labels[len(fg_inds_det):] = 0
            input_box_labels[len(fg_inds_det):] = 0
            seg_overlap = roidb['seg_overlap'][keep_inds]
            bg_inds_seg = np.where(seg_overlap < cfg.TRAIN.FG_SEG_THRESH)[0]
            input_mask_labels = input_box_labels.copy()
            input_mask_labels[bg_inds_seg] = 0

            gt_classes = roidb['gt_classes']
            input_masks = roidb['masks'][keep_inds, :, :]
            input_boxes = roidb['boxes'][keep_inds, :] * im_scales[im_i]

            mask_target = roidb['mask_targets']
            mask_target = mask_target[keep_inds, :, :]
            mask_resize = np.zeros((input_masks.shape[0], self.input_mz, self.input_mz))
            for i in xrange(mask_target.shape[0]):
                mask_resize[i, :, :] = cv2.resize(input_masks[i, :, :].astype(np.float), (self.input_mz, self.input_mz))
            mask_resize = mask_resize >= cfg.BINARIZE_THRESH

            mask_target_weights = np.zeros(mask_target.shape)
            
            mask_target_weights[0:len(fg_inds_det), :, :] = 1

            gt_boxes = roidb['boxes'][0:num_gt, :] * im_scales[im_i]
            gt_assignment = roidb['gt_assignment'][:, 0]
            bbox_target_data = bbox_compute_targets(input_boxes, gt_boxes[gt_assignment[keep_inds], :4], False)
            # normalize targets
            bbox_target_data = np.hstack((input_box_labels, bbox_target_data))\
                .astype(np.float32, copy=False)
            bbox_targets, bbox_inside_weights = get_bbox_regression_label(
                bbox_target_data, self._num_classes)

            for i in xrange(len(fg_inds_det)):
                cls = gt_classes[gt_assignment[fg_inds_det[i]]][0]
                if cls == 0:
                    continue
                mean = self._mean
                std = self._std
                bbox_targets[i, cls*4:cls*4+4] -= mean[cls, :]
                bbox_targets[i, cls*4:cls*4+4] /= std[cls, :]

            bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
            input_boxes = np.hstack((im_i * np.ones((input_boxes.shape[0], 1)), input_boxes))
            bz = input_boxes.shape[0]
            rois_blob = np.vstack((rois_blob, input_boxes))
            masks_blob = np.concatenate((masks_blob,
                                         mask_resize.reshape(bz, 1, self.input_mz, self.input_mz)), axis=0)
            box_labels_blob = np.concatenate((box_labels_blob, input_box_labels), axis=0)
            mask_labels_blob = np.concatenate((mask_labels_blob, input_mask_labels), axis=0)
            bbox_targets_blob = np.concatenate((bbox_targets_blob, bbox_targets), axis=0)
            mask_targets_blob = np.concatenate((mask_targets_blob,
                                                mask_target.reshape(bz, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)), axis=0)
            bbox_inside_weights_blob = np.concatenate((bbox_inside_weights_blob, bbox_inside_weights), axis=0)
            bbox_outside_weights_blob = np.concatenate((bbox_outside_weights_blob, bbox_outside_weights), axis=0)
            mask_weights_blob = np.concatenate((mask_weights_blob,
                                                mask_target_weights.reshape(bz, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)), axis=0)

        return {
            'data': im_blob,
            'rois': rois_blob,
            'masks': masks_blob,
            'box_label': box_labels_blob,
            'mask_label': mask_labels_blob,
            'bbox_targets': bbox_targets_blob,
            'mask_targets': mask_targets_blob,
            'bbox_inside_weights': bbox_inside_weights_blob,
            'bbox_outside_weights': bbox_outside_weights_blob,
            'mask_weight': mask_weights_blob
        }
