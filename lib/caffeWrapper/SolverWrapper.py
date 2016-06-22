# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import os
import numpy as np

from utils.timer import Timer
from mnc_config import cfg
from db.roidb import add_bbox_regression_targets, compute_mcg_mean_std
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2


class SolverWrapper(object):
    """ A simple wrapper around Caffe's solver.
        This wrapper gives us control over he snapshotting process, which we
        use to unnormalize the learned bounding-box regression weights.
    """
    def __init__(self, solver_prototxt, roidb, maskdb, output_dir, imdb,
                 pretrained_model=None):
        self.output_dir = output_dir
        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
                cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            if not cfg.CFM_MODE:
                print 'Computing bounding-box regression targets...'
                self.bbox_means, self.bbox_stds = add_bbox_regression_targets(roidb)
                print 'done'
            else:
                # Pre-defined mcg bbox_mean and bbox_std
                # We store them on disk to avoid disk level IO
                # multiple times (mcg boxes are stored on disk)
                mean_cache = './data/cache/mcg_bbox_mean.npy'
                std_cache = './data/cache/mcg_bbox_std.npy'
                roidb_dir = imdb._roidb_path
                if os.path.exists(mean_cache) and os.path.exists(std_cache):
                    self.bbox_means = np.load(mean_cache)
                    self.bbox_stds = np.load(std_cache)
                else:
                    self.bbox_means, self.bbox_stds = compute_mcg_mean_std(roidb_dir, imdb.num_classes)

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print 'Loading pretrained model weights from {:s}'.format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)
        if not cfg.CFM_MODE:
            self.solver.net.layers[0].set_roidb(roidb)
            if cfg.MNC_MODE:
                self.solver.net.layers[0].set_maskdb(maskdb)
        else:
            self.solver.net.layers[0].set_image_info(imdb, self.bbox_means, self.bbox_stds)

    def snapshot(self):
        """ Take a snapshot of the network after unnormalizing the learned
            bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net
        # I'm wondering whether I still need to keep it if only faster-RCNN is needed
        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             'bbox_pred' in net.params)
        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()
            if cfg.CFM_MODE:
                cfm_mean = self.bbox_means.ravel()
                cfm_std = self.bbox_stds.ravel()
                net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data * cfm_std[:, np.newaxis])
                net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data * cfm_std + cfm_mean)
            else:
                # scale and shift with transform reg unnormalization; then save snapshot
                net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
                net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # If we specify an infix in the configuration
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')

        # For snapshot caffemodel, since MNC use shared parameters
        # but caffe save parameters according to layer name instead of
        # parameter names, its size will exceed 2GB, which make program crash
        # Luckily, we may save it to HDF5 to avoid this issues
        if not cfg.MNC_MODE:
            filename = os.path.join(self.output_dir, filename)
            net.save(str(filename))
        else:
            filename = os.path.join(self.output_dir, filename + '.h5')
            net.save_to_hdf5(str(filename), False)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1

    def train_model(self, max_iters):
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

