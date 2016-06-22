# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import cPickle
import os
import scipy.io as sio
import numpy as np
from datasets.pascal_voc_det import PascalVOCDet
from mnc_config import cfg
from utils.vis_seg import vis_seg
from utils.voc_eval import voc_eval_sds
import scipy


class PascalVOCSeg(PascalVOCDet):
    """
    A subclass for datasets.imdb.imdb
    This class contains information of ROIDB and MaskDB
    This class implements roidb and maskdb related functions
    """
    def __init__(self, image_set, year, devkit_path=None):
        PascalVOCDet.__init__(self, image_set, year, devkit_path)
        self._ori_image_num = len(self._image_index)
        self._comp_id = 'comp6'
        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}
        self._data_path = os.path.join(self._devkit_path)
        self._roidb_path = os.path.join(self.cache_path, 'voc_2012_' + image_set + '_mcg_maskdb')

    def image_path_at(self, i):
        image_path = os.path.join(self._data_path, 'img', self._image_index[i] + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def roidb_path_at(self, i):
        if i >= self._ori_image_num:
            return os.path.join(self._roidb_path,
                                self.image_index[i % self._ori_image_num] + '_flip.mat')
        else:
            return os.path.join(self._roidb_path,
                                self.image_index[i] + '.mat')

    def gt_maskdb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_maskdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_maskdb = cPickle.load(fid)
            print '{} gt maskdb loaded from {}'.format(self.name, cache_file)
        else:
            num_image = len(self.image_index)
            gt_roidbs = self.gt_roidb()
            gt_maskdb = [self._load_sbd_mask_annotations(index, gt_roidbs)
                         for index in xrange(num_image)]
            with open(cache_file, 'wb') as fid:
                cPickle.dump(gt_maskdb, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote gt roidb to {}'.format(cache_file)
        return gt_maskdb

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_sbd_mask_annotations(self, index, gt_roidbs):
        """
        Load gt_masks information from SBD's additional data
        """
        if index % 1000 == 0:
            print '%d / %d' % (index, len(self._image_index))
        image_name = self._image_index[index]
        inst_file_name = os.path.join(self._data_path, 'inst', image_name + '.mat')
        gt_inst_mat = scipy.io.loadmat(inst_file_name)
        gt_inst_data = gt_inst_mat['GTinst']['Segmentation'][0][0]
        unique_inst = np.unique(gt_inst_data)
        background_ind = np.where(unique_inst == 0)[0]
        unique_inst = np.delete(unique_inst, background_ind)
        gt_roidb = gt_roidbs[index]
        cls_file_name = os.path.join(self._data_path, 'cls', image_name + '.mat')
        gt_cls_mat = scipy.io.loadmat(cls_file_name)
        gt_cls_data = gt_cls_mat['GTcls']['Segmentation'][0][0]
        gt_masks = []
        for ind, inst_mask in enumerate(unique_inst):
            box = gt_roidb['boxes'][ind]
            im_mask = (gt_inst_data == inst_mask)
            im_cls_mask = np.multiply(gt_cls_data, im_mask)
            unique_cls_inst = np.unique(im_cls_mask)
            background_ind = np.where(unique_cls_inst == 0)[0]
            unique_cls_inst = np.delete(unique_cls_inst, background_ind)
            assert len(unique_cls_inst) == 1
            assert unique_cls_inst[0] == gt_roidb['gt_classes'][ind]
            mask = im_mask[box[1]: box[3]+1, box[0]:box[2]+1]
            gt_masks.append(mask)

        # Also record the maximum dimension to create fixed dimension array when do forwarding
        mask_max_x = max(gt_masks[i].shape[1] for i in xrange(len(gt_masks)))
        mask_max_y = max(gt_masks[i].shape[0] for i in xrange(len(gt_masks)))
        return {
            'gt_masks': gt_masks,
            'mask_max': [mask_max_x, mask_max_y],
            'flipped': False
        }

    def append_flipped_masks(self):
        """
        This method is only accessed when we use maskdb, so implement here
        Append flipped images to mask database
        Note this method doesn't actually flip the 'image', it flip masks instead
        """
        cache_file = os.path.join(self.cache_path, self.name + '_' + cfg.TRAIN.PROPOSAL_METHOD + '_maskdb_flip.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                flip_maskdb = cPickle.load(fid)
            print '{} gt flipped roidb loaded from {}'.format(self.name, cache_file)
            self.maskdb.extend(flip_maskdb)
            # Need to check this condition since otherwise we may occasionally *4
            if self._image_index == self.num_images:
                self._image_index *= 2
        else:
            # pure image number hold for future development
            # this is useless since append flip mask will only be called once
            num_images = self._ori_image_num
            flip_maskdb = []
            for i in xrange(num_images):
                masks = self.maskdb[i]['gt_masks']
                masks_flip = []
                for mask_ind in xrange(len(masks)):
                    mask_flip = np.fliplr(masks[mask_ind])
                    masks_flip.append(mask_flip)
                entry = {'gt_masks': masks_flip,
                         'mask_max': self.maskdb[i]['mask_max'],
                         'flipped': True}
                flip_maskdb.append(entry)
            with open(cache_file, 'wb') as fid:
                cPickle.dump(flip_maskdb, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote gt flipped maskdb to {}'.format(cache_file)
            self.maskdb.extend(flip_maskdb)
            # Need to check this condition since otherwise we may occasionally *4
            if self._image_index == self.num_images:
                self._image_index *= 2

    def visualization_segmentation(self, output_dir):
        vis_seg(self.image_index, self.classes, output_dir, self._data_path)

    # --------------------------- Evaluation ---------------------------
    def evaluate_segmentation(self, all_boxes, all_masks, output_dir):
        self._write_voc_seg_results_file(all_boxes, all_masks, output_dir)
        self._py_evaluate_segmentation(output_dir)

    def _write_voc_seg_results_file(self, all_boxes, all_masks, output_dir):
        """
        Write results as a pkl file, note this is different from
        detection task since it's difficult to write masks to txt
        """
        # Always reformat result in case of sometimes masks are not
        # binary or is in shape (n, sz*sz) instead of (n, sz, sz)
        all_boxes, all_masks = self._reformat_result(all_boxes, all_masks)
        for cls_inds, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = os.path.join(output_dir, cls + '_det.pkl')
            with open(filename, 'wr') as f:
                cPickle.dump(all_boxes[cls_inds], f, cPickle.HIGHEST_PROTOCOL)
            filename = os.path.join(output_dir, cls + '_seg.pkl')
            with open(filename, 'wr') as f:
                cPickle.dump(all_masks[cls_inds], f, cPickle.HIGHEST_PROTOCOL)

    def _reformat_result(self, boxes, masks):
        num_images = len(self.image_index)
        num_class = len(self.classes)
        reformat_masks = [[[] for _ in xrange(num_images)]
                          for _ in xrange(num_class)]
        for cls_inds in xrange(1, num_class):
            for img_inds in xrange(num_images):
                if len(masks[cls_inds][img_inds]) == 0:
                    continue
                num_inst = masks[cls_inds][img_inds].shape[0]
                reformat_masks[cls_inds][img_inds] = masks[cls_inds][img_inds]\
                    .reshape(num_inst, cfg.MASK_SIZE, cfg.MASK_SIZE)
                reformat_masks[cls_inds][img_inds] = reformat_masks[cls_inds][img_inds] >= cfg.BINARIZE_THRESH
        all_masks = reformat_masks
        return boxes, all_masks

    def _py_evaluate_segmentation(self, output_dir):
        gt_dir = self._data_path
        imageset_file = os.path.join(gt_dir, self._image_set + '.txt')
        cache_dir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # define this as true according to SDS's evaluation protocol
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        print '~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~'
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, gt_dir,
                              imageset_file, cls, cache_dir, self._classes, ov_thresh=0.5)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
        print('Mean AP@0.5 = {:.2f}'.format(np.mean(aps)*100))
        print '~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~'
        aps = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, gt_dir,
                              imageset_file, cls, cache_dir, self._classes, ov_thresh=0.7)
            aps += [ap]
            print('AP for {} = {:.2f}'.format(cls, ap*100))
        print('Mean AP@0.7 = {:.2f}'.format(np.mean(aps)*100))

