# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import numpy as np
import scipy.sparse
from mnc_config import cfg


class PascalVOC(object):
    """ A base class for image database."""
    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        self._maskdb = None
        self._maskdb_handler = self.default_maskdb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def maskdb_handler(self):
        return self._roidb_handler

    @maskdb_handler.setter
    def maskdb_handler(self, val):
        self._roidb_handler = val

    @property
    def roidb(self):
        # A roidb is a 'list of dictionaries', each with the following keys:
        #   boxes: the numpy array for boxes coordinate
        #   gt_overlaps: overlap ratio for ground truth
        #   gt_classes: ground truth class for that box
        #   flipped: whether get flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def maskdb(self):
        if self._maskdb is not None:
            return self._maskdb
        else:
            self._maskdb = self.maskdb_handler()
            return self._maskdb

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def set_roi_handler(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    def set_mask_handler(self, method):
        method = eval('self.' + method + '_maskdb')
        self.maskdb_handler = method

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def default_maskdb(self):
        raise NotImplementedError

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
        return a
