# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import uuid
import cPickle
import numpy as np
import scipy.sparse
import PIL

import xml.etree.ElementTree as xmlET
from datasets.pascal_voc import PascalVOC
from mnc_config import cfg
from utils.voc_eval import voc_eval


class PascalVOCDet(PascalVOC):
    """
    A subclass for PascalVOC
    """
    def __init__(self, image_set, year, devkit_path=None):
        PascalVOC.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year) if 'SDS' not in self._devkit_path else self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        num_image = len(self.image_index)
        if cfg.MNC_MODE:
            gt_roidb = [self._load_sbd_annotations(index) for index in xrange(num_image)]
        else:
            gt_roidb = [self._load_pascal_annotations(index) for index in xrange(num_image)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        Examples
        path is: self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        --------
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def append_flipped_rois(self):
        """
        This method is irrelevant with database, so implement here
        Append flipped images to ROI database
        Note this method doesn't actually flip the 'image', it flip
        boxes instead
        """
        cache_file = os.path.join(self.cache_path, self.name + '_' + cfg.TRAIN.PROPOSAL_METHOD + '_roidb_flip.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                flip_roidb = cPickle.load(fid)
            print '{} gt flipped roidb loaded from {}'.format(self.name, cache_file)
        else:
            num_images = self.num_images
            widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                      for i in xrange(num_images)]
            flip_roidb = []
            for i in xrange(num_images):
                boxes = self.roidb[i]['boxes'].copy()
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = widths[i] - oldx2 - 1
                boxes[:, 2] = widths[i] - oldx1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                entry = {'boxes': boxes,
                         'gt_overlaps': self.roidb[i]['gt_overlaps'],
                         'gt_classes': self.roidb[i]['gt_classes'],
                         'flipped': True}
                flip_roidb.append(entry)
            with open(cache_file, 'wb') as fid:
                cPickle.dump(flip_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote gt flipped roidb to {}'.format(cache_file)

        self.roidb.extend(flip_roidb)
        self._image_index *= 2

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def _load_pascal_annotations(self, index):
        """
        Load image and bounding boxes info from XML file
        in the PASCAL VOC format according to image index
        """
        image_name = self._image_index[index]
        filename = os.path.join(self._data_path, 'Annotations', image_name + '.xml')
        tree = xmlET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            if len(non_diff_objs) != len(objs):
                print 'Removed {} difficult objects'.format(len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        # boxes[ind, :] will be boxes
        # gt_classes[ind] will be the associated class name for this box
        # overlaps[ind, class] will assign 1.0 to ground truth
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def _load_sbd_annotations(self, index):
        if index % 1000 == 0: print '%d / %d' % (index, len(self._image_index))
        image_name = self._image_index[index]
        inst_file_name = os.path.join(self._data_path, 'inst', image_name + '.mat')
        gt_inst_mat = scipy.io.loadmat(inst_file_name)
        gt_inst_data = gt_inst_mat['GTinst']['Segmentation'][0][0]
        unique_inst = np.unique(gt_inst_data)
        background_ind = np.where(unique_inst == 0)[0]
        unique_inst = np.delete(unique_inst, background_ind)

        cls_file_name = os.path.join(self._data_path, 'cls', image_name + '.mat')
        gt_cls_mat = scipy.io.loadmat(cls_file_name)
        gt_cls_data = gt_cls_mat['GTcls']['Segmentation'][0][0]

        boxes = np.zeros((len(unique_inst), 4), dtype=np.uint16)
        gt_classes = np.zeros(len(unique_inst), dtype=np.int32)
        overlaps = np.zeros((len(unique_inst), self.num_classes), dtype=np.float32)
        for ind, inst_mask in enumerate(unique_inst):
            im_mask = (gt_inst_data == inst_mask)
            im_cls_mask = np.multiply(gt_cls_data, im_mask)
            unique_cls_inst = np.unique(im_cls_mask)
            background_ind = np.where(unique_cls_inst == 0)[0]
            unique_cls_inst = np.delete(unique_cls_inst, background_ind)
            assert len(unique_cls_inst) == 1
            gt_classes[ind] = unique_cls_inst[0]
            [r, c] = np.where(im_mask > 0)
            boxes[ind, 0] = np.min(c)
            boxes[ind, 1] = np.min(r)
            boxes[ind, 2] = np.max(c)
            boxes[ind, 3] = np.max(r)
            overlaps[ind, unique_cls_inst[0]] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    """-----------------Evaluation--------------------"""
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            raise NotImplementedError
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        print '--------------------------------------------------------------'
        print 'Computing results with **unofficial** Python eval code.'
        print 'Results should be very close to the official MATLAB eval code.'
        print 'Recompute with `./tools/reval.py --matlab ...` for your paper.'
        print '--------------------------------------------------------------'
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')

