# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import cPickle
import scipy
import numpy as np
import cv2
import heapq

import caffe
from utils.timer import Timer
from nms.nms_wrapper import apply_nms, apply_nms_mask_single
from mnc_config import cfg, get_output_dir
from utils.blob import prep_im_for_blob, im_list_to_blob, prep_im_for_blob_cfm, pred_rois_for_blob
from transform.bbox_transform import clip_boxes, bbox_transform_inv, filter_small_boxes
from transform.mask_transform import cpu_mask_voting, gpu_mask_voting


class TesterWrapper(object):
    """
    A simple wrapper around Caffe's test forward
    """
    def __init__(self, test_prototxt, imdb, test_model, task_name):
        # Pre-processing, test whether model stored in binary file or npy files
        self.net = caffe.Net(test_prototxt, test_model, caffe.TEST)
        self.net.name = os.path.splitext(os.path.basename(test_model))[0]
        self.imdb = imdb
        self.output_dir = get_output_dir(imdb, self.net)
        self.task_name = task_name
        # We define some class variables here to avoid defining them many times in every method
        self.num_images = len(self.imdb.image_index)
        self.num_classes = self.imdb.num_classes
        # heuristic: keep an average of 40 detections per class per images prior to nms
        self.max_per_set = 40 * self.num_images
        # heuristic: keep at most 100 detection per class per image prior to NMS
        self.max_per_image = 100

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_result(self):
        output_dir = self.output_dir
        det_file = os.path.join(output_dir, 'res_boxes.pkl')
        seg_file = os.path.join(output_dir, 'res_masks.pkl')
        if self.task_name == 'det':
            self.get_detection_result()
        elif self.task_name == 'vis_seg':
            self.vis_segmentation_result()
        elif self.task_name == 'seg':
            if os.path.isfile(det_file) and os.path.isfile(seg_file):
                with open(det_file, 'rb') as f:
                    seg_box = cPickle.load(f)
                with open(seg_file, 'rb') as f:
                    seg_mask = cPickle.load(f)
            else:
                seg_box, seg_mask = self.get_segmentation_result()
                with open(det_file, 'wb') as f:
                    cPickle.dump(seg_box, f, cPickle.HIGHEST_PROTOCOL)
                with open(seg_file, 'wb') as f:
                    cPickle.dump(seg_mask, f, cPickle.HIGHEST_PROTOCOL)
            print 'Evaluating segmentation using MNC 5 stage inference'
            self.imdb.evaluate_segmentation(seg_box, seg_mask, output_dir)
        elif self.task_name == 'cfm':
            if os.path.isfile(det_file) and os.path.isfile(seg_file):
                with open(det_file, 'rb') as f:
                    cfm_boxes = cPickle.load(f)
                with open(seg_file, 'rb') as f:
                    cfm_masks = cPickle.load(f)
            else:
                cfm_boxes, cfm_masks = self.get_cfm_result()
                with open(det_file, 'wb') as f:
                    cPickle.dump(cfm_boxes, f, cPickle.HIGHEST_PROTOCOL)
                with open(seg_file, 'wb') as f:
                    cPickle.dump(cfm_masks, f, cPickle.HIGHEST_PROTOCOL)
            print 'Evaluating segmentation using convolutional feature masking'
            self.imdb.evaluate_segmentation(cfm_boxes, cfm_masks, output_dir)
        else:
            print 'task name only support \'det\', \'seg\', \'cfm\' and \'vis_seg\''
            raise NotImplementedError

    def get_detection_result(self):
        output_dir = self.output_dir
        # heuristic: keep an average of 40 detections per class per images prior to NMS
        max_per_set = 40 * self.num_images
        # heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = 100
        # detection threshold for each class (this is adaptively set based on the
        # max_per_set constraint)
        thresh = -np.inf * np.ones(self.num_classes)
        # top_scores will hold one min heap of scores per class (used to enforce
        # the max_per_set constraint)
        top_scores = [[] for _ in xrange(self.num_classes)]
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.num_classes)]
        _t = {'im_detect': Timer(), 'misc': Timer()}
        for i in xrange(self.num_images):
            im = cv2.imread(self.imdb.image_path_at(i))
            _t['im_detect'].tic()
            scores, boxes = self._detection_forward(im)
            _t['im_detect'].toc()
            for j in xrange(1, self.num_classes):
                inds = np.where(scores[:, j] > thresh[j])[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                top_inds = np.argsort(-cls_scores)[:max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                # push new scores onto the min heap
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the min heap and update the class threshold
                if len(top_scores[j]) > max_per_set:
                    while len(top_scores[j]) > max_per_set:
                        heapq.heappop(top_scores[j])
                    thresh[j] = top_scores[j][0]

                all_boxes[j][i] = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
                    .astype(np.float32, copy=False)
            print 'process image %d/%d, forward average time %f' % (i, self.num_images,
                                                                    _t['im_detect'].average_time)

        for j in xrange(1, self.num_classes):
            for i in xrange(self.num_images):
                inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
                all_boxes[j][i] = all_boxes[j][i][inds, :]

        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

        print 'Applying NMS to all detections'
        nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

        print 'Evaluating detections'
        self.imdb.evaluate_detections(nms_dets, output_dir)

    def vis_segmentation_result(self):
        self.imdb.visualization_segmentation(self.output_dir)

    def get_segmentation_result(self):
        # detection threshold for each class
        # (this is adaptively set based on the max_per_set constraint)
        thresh = -np.inf * np.ones(self.num_classes)
        # top_scores will hold one min heap of scores per class (used to enforce
        # the max_per_set constraint)
        top_scores = [[] for _ in xrange(self.num_classes)]
        # all detections and segmentation are collected into a list:
        # Since the number of dets/segs are of variable size
        all_boxes = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.num_classes)]
        all_masks = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        for i in xrange(self.num_images):
            im = cv2.imread(self.imdb.image_path_at(i))
            _t['im_detect'].tic()
            masks, boxes, seg_scores = self._segmentation_forward(im)
            _t['im_detect'].toc()
            if not cfg.TEST.USE_MASK_MERGE:
                for j in xrange(1, self.num_classes):
                    inds = np.where(seg_scores[:, j] > thresh[j])[0]
                    cls_scores = seg_scores[inds, j]
                    cls_boxes = boxes[inds, :]
                    cls_masks = masks[inds, :]
                    top_inds = np.argsort(-cls_scores)[:self.max_per_image]
                    cls_scores = cls_scores[top_inds]
                    cls_boxes = cls_boxes[top_inds, :]
                    cls_masks = cls_masks[top_inds, :]
                    # push new scores onto the min heap
                    for val in cls_scores:
                        heapq.heappush(top_scores[j], val)
                    # if we've collected more than the max number of detection,
                    # then pop items off the min heap and update the class threshold
                    if len(top_scores[j]) > self.max_per_set:
                        while len(top_scores[j]) > self.max_per_set:
                            heapq.heappop(top_scores[j])
                        thresh[j] = top_scores[j][0]
                    # Add new boxes into record
                    box_before_nms = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
                        .astype(np.float32, copy=False)
                    mask_before_nms = cls_masks.astype(np.float32, copy=False)
                    all_boxes[j][i], all_masks[j][i] = apply_nms_mask_single(box_before_nms, mask_before_nms, cfg.TEST.NMS)
            else:
                if cfg.TEST.USE_GPU_MASK_MERGE:
                    result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, self.num_classes,
                                                              self.max_per_image, im.shape[1], im.shape[0])
                else:
                    result_box, result_mask = cpu_mask_voting(masks, boxes, seg_scores, self.num_classes,
                                                              self.max_per_image, im.shape[1], im.shape[0])
                # no need to create a min heap since the output will not exceed max number of detection
                for j in xrange(1, self.num_classes):
                    all_boxes[j][i] = result_box[j-1]
                    all_masks[j][i] = result_mask[j-1]

            print 'process image %d/%d, forward average time %f' % (i, self.num_images,
                                                                    _t['im_detect'].average_time)

        for j in xrange(1, self.num_classes):
            for i in xrange(self.num_images):
                inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
                all_boxes[j][i] = all_boxes[j][i][inds, :]
                all_masks[j][i] = all_masks[j][i][inds]

        return all_boxes, all_masks

    def _detection_forward(self, im):
        """ Detect object classes in an image given object proposals.
        Arguments:
            im (ndarray): color image to test (in BGR order)
        Returns:
            box_scores (ndarray): R x K array of object class scores (K includes
                background as object category 0)
            all_boxes (ndarray): R x (4*K) array of predicted bounding boxes
        """
        forward_kwargs, im_scales = self._prepare_mnc_args(im)
        blobs_out = self.net.forward(**forward_kwargs)
        # There are some data we need to get:
        # 1. ROIS (with bbox regression)
        rois = self.net.blobs['rois'].data.copy()
        # un-scale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes, _ = clip_boxes(pred_boxes, im.shape)
        # 2. Detection score
        scores = blobs_out['cls_prob']
        return scores, pred_boxes

    def _segmentation_forward(self, im):
        forward_kwargs, im_scales = self._prepare_mnc_args(im)
        blobs_out = self.net.forward(**forward_kwargs)
        # output we need to collect:
        # 1. output from phase1'
        rois_phase1 = self.net.blobs['rois'].data.copy()
        masks_phase1 = self.net.blobs['mask_proposal'].data[...]
        scores_phase1 = self.net.blobs['seg_cls_prob'].data[...]
        # 2. output from phase2
        rois_phase2 = self.net.blobs['rois_ext'].data[...]
        masks_phase2 = self.net.blobs['mask_proposal_ext'].data[...]
        scores_phase2 = self.net.blobs['seg_cls_prob_ext'].data[...]
        # Boxes are in resized space, we un-scale them back
        rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
        rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
        rois_phase1, _ = clip_boxes(rois_phase1, im.shape)
        rois_phase2, _ = clip_boxes(rois_phase2, im.shape)
        # concatenate two stages to get final network output
        masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
        boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
        scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
        return masks, boxes, scores

    def _prepare_mnc_args(self, im):
        # Prepare image data blob
        blobs = {'data': None}
        processed_ims = []
        im, im_scale_factors = \
            prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
        processed_ims.append(im)
        blobs['data'] = im_list_to_blob(processed_ims)
        # Prepare image info blob
        im_scales = [np.array(im_scale_factors)]
        assert len(im_scales) == 1, 'Only single-image batch implemented'
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
        # Reshape network inputs and do forward
        self.net.blobs['data'].reshape(*blobs['data'].shape)
        self.net.blobs['im_info'].reshape(*blobs['im_info'].shape)
        forward_kwargs = {
            'data': blobs['data'].astype(np.float32, copy=False),
            'im_info': blobs['im_info'].astype(np.float32, copy=False)
        }
        return forward_kwargs, im_scales

    def get_cfm_result(self):
        # detection threshold for each class
        # (this is adaptively set based on the max_per_set constraint)
        thresh = -np.inf * np.ones(self.num_classes)
        # top_scores will hold one min heap of scores per class (used to enforce
        # the max_per_set constraint)
        top_scores = [[] for _ in xrange(self.num_classes)]
        # all detections and segmentation are collected into a list:
        # Since the number of dets/segs are of variable size
        all_boxes = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.num_classes)]
        all_masks = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.num_classes)]
        _t = {'im_detect': Timer(), 'misc': Timer()}
        for i in xrange(self.num_images):
            _t['im_detect'].tic()
            masks, boxes, seg_scores = self.cfm_network_forward(i)
            for j in xrange(1, self.num_classes):
                inds = np.where(seg_scores[:, j] > thresh[j])[0]
                cls_scores = seg_scores[inds, j]
                cls_boxes = boxes[inds, :]
                cls_masks = masks[inds, :]
                top_inds = np.argsort(-cls_scores)[:self.max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                cls_masks = cls_masks[top_inds, :]
                # push new scores onto the min heap
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the min heap and update the class threshold
                if len(top_scores[j]) > self.max_per_set:
                    while len(top_scores[j]) > self.max_per_set:
                        heapq.heappop(top_scores[j])
                    thresh[j] = top_scores[j][0]
                box_before_nms = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
                    .astype(np.float32, copy=False)
                mask_before_nms = cls_masks.astype(np.float32, copy=False)
                all_boxes[j][i], all_masks[j][i] = apply_nms_mask_single(box_before_nms, mask_before_nms, cfg.TEST.NMS)
            _t['im_detect'].toc()
            print 'process image %d/%d, forward average time %f' % (i, self.num_images,
                                                                    _t['im_detect'].average_time)
        for j in xrange(1, self.num_classes):
            for i in xrange(self.num_images):
                inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
                all_boxes[j][i] = all_boxes[j][i][inds, :]
                all_masks[j][i] = all_masks[j][i][inds]

        return all_boxes, all_masks

    def cfm_network_forward(self, im_i):
        im = cv2.imread(self.imdb.image_path_at(im_i))
        roidb_cache = os.path.join('data/cache/voc_2012_val_mcg_maskdb/', self.imdb._image_index[im_i] + '.mat')
        roidb = scipy.io.loadmat(roidb_cache)
        boxes = roidb['boxes']
        filter_keep = filter_small_boxes(boxes, min_size=16)
        boxes = boxes[filter_keep, :]
        masks = roidb['masks']
        masks = masks[filter_keep, :, :]
        assert boxes.shape[0] == masks.shape[0]

        # Resize input mask, make it the same as CFM's input size
        mask_resize = np.zeros((masks.shape[0], cfg.TEST.CFM_INPUT_MASK_SIZE, cfg.TEST.CFM_INPUT_MASK_SIZE))
        for i in xrange(masks.shape[0]):
            mask_resize[i, :, :] = cv2.resize(masks[i, :, :].astype(np.float),
                                              (cfg.TEST.CFM_INPUT_MASK_SIZE, cfg.TEST.CFM_INPUT_MASK_SIZE))
        masks = mask_resize

        # Get top-k proposals from MCG
        if cfg.TEST.USE_TOP_K_MCG:
            num_keep = min(boxes.shape[0], cfg.TEST.USE_TOP_K_MCG)
            boxes = boxes[:num_keep, :]
            masks = masks[:num_keep, :, :]
            assert boxes.shape[0] == masks.shape[0]
        # deal with multi-scale test
        # we group several adjacent scales to do forward
        _, im_scale_factors = prep_im_for_blob_cfm(im, cfg.TEST.SCALES)
        orig_boxes = boxes.copy()
        boxes = pred_rois_for_blob(boxes, im_scale_factors)
        num_scale_iter = int(np.ceil(len(cfg.TEST.SCALES) / float(cfg.TEST.GROUP_SCALE)))
        LO_SCALE = 0
        MAX_ROIS_GPU = cfg.TEST.MAX_ROIS_GPU
        # set up return results
        res_boxes = np.zeros((0, 4), dtype=np.float32)
        res_masks = np.zeros((0, 1, cfg.MASK_SIZE, cfg.MASK_SIZE), dtype=np.float32)
        res_seg_scores = np.zeros((0, self.num_classes), dtype=np.float32)

        for scale_iter in xrange(num_scale_iter):
            HI_SCALE = min(LO_SCALE + cfg.TEST.GROUP_SCALE, len(cfg.TEST.SCALES))
            inds_this_scale = np.where((boxes[:, 0] >= LO_SCALE) & (boxes[:, 0] < HI_SCALE))[0]
            if len(inds_this_scale) == 0:
                LO_SCALE += cfg.TEST.GROUP_SCALE
                continue
            max_rois_this_scale = MAX_ROIS_GPU[scale_iter]
            boxes_this_scale = boxes[inds_this_scale, :]
            masks_this_scale = masks[inds_this_scale, :, :]
            num_iter_this_scale = int(np.ceil(boxes_this_scale.shape[0] / float(max_rois_this_scale)))
            # make the batch index of input box start from 0
            boxes_this_scale[:, 0] -= min(boxes_this_scale[:, 0])
            # re-prepare im blob for this_scale
            input_blobs = {}
            input_blobs['data'], _ = prep_im_for_blob_cfm(im, cfg.TEST.SCALES[LO_SCALE:HI_SCALE])
            input_blobs['data'] = input_blobs['data'].astype(np.float32, copy=False)
            input_start = 0
            for test_iter in xrange(num_iter_this_scale):
                input_end = min(input_start + max_rois_this_scale, boxes_this_scale.shape[0])
                input_box = boxes_this_scale[input_start:input_end, :]
                input_mask = masks_this_scale[input_start:input_end, :, :]
                input_blobs['rois'] = input_box.astype(np.float32, copy=False)
                input_blobs['masks'] = input_mask.reshape(input_box.shape[0], 1,
                                                    cfg.TEST.CFM_INPUT_MASK_SIZE, cfg.TEST.CFM_INPUT_MASK_SIZE
                                                    ).astype(np.float32, copy=False)
                input_blobs['masks'] = (input_blobs['masks'] >= cfg.BINARIZE_THRESH).astype(np.float32, copy=False)
                self.net.blobs['data'].reshape(*input_blobs['data'].shape)
                self.net.blobs['rois'].reshape(*input_blobs['rois'].shape)
                self.net.blobs['masks'].reshape(*input_blobs['masks'].shape)
                blobs_out = self.net.forward(**input_blobs)
                output_mask = blobs_out['mask_prob'].copy()
                output_score = blobs_out['seg_cls_prob'].copy()
                res_masks = np.vstack((res_masks,
                                       output_mask.reshape(
                                           input_box.shape[0], 1, cfg.MASK_SIZE, cfg.MASK_SIZE
                                       ).astype(np.float32, copy=False)))
                res_seg_scores = np.vstack((res_seg_scores, output_score))
                input_start += max_rois_this_scale
            res_boxes = np.vstack((res_boxes, orig_boxes[inds_this_scale, :]))
            LO_SCALE += cfg.TEST.GROUP_SCALE

        return res_masks, res_boxes, res_seg_scores
