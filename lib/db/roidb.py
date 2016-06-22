# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import PIL
import numpy as np
import os
import cPickle
import scipy

from db.imdb import get_imdb
from mnc_config import cfg
from transform.bbox_transform import compute_targets


def prepare_roidb(imdb):
    """ Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function pre-computes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = \
            compute_targets(rois, max_overlaps, max_classes)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Use fixed / precomputed "means" and "stds" instead of empirical values
        means = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1))
        stds = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes, 1))
    else:
        # Compute values needed for means and stds
        # var(x) = E(x^2) - E(x)^2
        class_counts = np.zeros((num_classes, 1)) + cfg.EPS
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                if cls_inds.size > 0:
                    class_counts[cls] += cls_inds.size
                    sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                    squared_sums[cls, :] += \
                            (targets[cls_inds, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print 'bbox target means:'
    print means
    print means[1:, :].mean(axis=0)  # ignore bg class
    print 'bbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0)  # ignore bg class

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        print "Normalizing targets"
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
    else:
        print "NOT normalizing targets"

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()


def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    # Here set handler function. (e.g. gt_roidb in faster RCNN)
    imdb.set_roi_handler(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_rois()
        print 'done'
    print 'Preparing training data...'
    prepare_roidb(imdb)
    print 'done'
    return imdb.roidb


def attach_roidb(imdb_names):
    """
    only implement single roidb now
    """
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        raise NotImplementedError
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


def compute_mcg_mean_std(roidb_dir, num_classes):
    """
    Compute bbox mean and stds for mcg proposals
    Since mcg proposal are stored on disk, so we precomputed it here once
    and save them to disk to avoid disk I/O next time
    Args:
        roidb_dir: directory contain all the mcg proposals
    """
    file_list = sorted(os.listdir(roidb_dir))
    target_list = []
    cnt = 0
    for file_name in file_list:
        roidb_cache = os.path.join(roidb_dir, file_name)
        roidb = scipy.io.loadmat(roidb_cache)
        target_list.append(compute_targets(roidb['boxes'], roidb['det_overlap'], roidb['output_label'].ravel()))
        cnt += 1

    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 4))
    squared_sums = np.zeros((num_classes, 4))
    for im_i in xrange(len(target_list)):
        targets = target_list[im_i]
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                squared_sums[cls, :] += \
                    (targets[cls_inds, 1:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)
    np.save('data/cache/mcg_bbox_mean.npy', means)
    np.save('data/cache/mcg_bbox_std.npy', stds)
    return means, stds
