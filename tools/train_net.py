#!/usr/bin/env python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import argparse
import sys
import pprint
import numpy as np
# User-defined module
import _init_paths
from mnc_config import cfg, cfg_from_file, get_output_dir  # config mnc
from db.roidb import attach_roidb
from db.maskdb import attach_maskdb
from caffeWrapper.SolverWrapper import SolverWrapper
import caffe


def parse_args():
    """ Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # get imdb and roidb from specified imdb_name
    imdb, roidb = attach_roidb(args.imdb_name)
    # Faster RCNN doesn't need
    if cfg.MNC_MODE or cfg.CFM_MODE:
        imdb, maskdb = attach_maskdb(args.imdb_name)
    else:
        maskdb = None
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    _solver = SolverWrapper(args.solver, roidb, maskdb, output_dir, imdb,
                            pretrained_model=args.pretrained_model)

    print 'Solving...'
    _solver.train_model(args.max_iters)
    print 'done solving'

