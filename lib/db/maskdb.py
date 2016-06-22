# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from mnc_config import cfg
from db.imdb import get_imdb


def get_maskdb(imdb_name):

    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    # Here set handler function. (e.g. gt_roidb in faster RCNN)
    imdb.set_roi_handler(cfg.TRAIN.PROPOSAL_METHOD)
    imdb.set_mask_handler(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_masks()
        print 'done'
    return imdb.maskdb


def attach_maskdb(imdb_names):
    """
    only implement single maskdb now
    """
    maskdbs = [get_maskdb(s) for s in imdb_names.split('+')]
    maskdb = maskdbs[0]
    if len(maskdbs) > 1:
        raise NotImplementedError
    else:
        imdb = get_imdb(imdb_names)
    return imdb, maskdb
