
"""MNC config system
"""
import numpy as np
import os.path
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# MNC/CFM mode
__C.MNC_MODE = True
__C.CFM_MODE = False

__C.EXP_DIR = 'default'
__C.USE_GPU_NMS = True
__C.GPU_ID = 0
__C.RNG_SEED = 3
__C.EPS = 1e-14
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Data directory
__C.DATA_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'data'))
# Related to mask resizing and binarize predicted masks
__C.BINARIZE_THRESH = 0.4
# Mask estimation (if any) size (may be different from CFM input size)
__C.MASK_SIZE = 21

# Training options
__C.TRAIN = edict()

# ------- General setting ----
__C.TRAIN.IMS_PER_BATCH = 1
# Batch size for training Region CNN (not RPN)
__C.TRAIN.BATCH_SIZE = 64
# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True
# Use flipped image for augmentation
__C.TRAIN.USE_FLIPPED = True
# Resize shortest side to 600
__C.TRAIN.SCALES = (600,)
__C.TRAIN.MAX_SIZE = 1000
__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.SNAPSHOT_INFIX = ''
# Sample FG
__C.TRAIN.FG_FRACTION = [0.3]
__C.TRAIN.FG_THRESH_HI = [1.0]
__C.TRAIN.FG_THRESH_LO = [0.5]
# Sample BF according to remaining samples
__C.TRAIN.BG_FRACTION = [0.85, 0.15]
__C.TRAIN.BG_THRESH_HI = [0.5, 0.1]
__C.TRAIN.BG_THRESH_LO = [0.1, 0.0]

# ------- Proposal -------
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# ------- BBOX Regression ---------
# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_THRESH = 0.5
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# weight of smooth L1 loss
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# -------- RPN ----------
# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IO < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor satisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
# Note this is class-agnostic anchors' FG_FRACTION
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# Mix anchors used for RPN and later layer
__C.TRAIN.MIX_INDEX = True

# -------- CFM ----------
__C.TRAIN.CFM_INPUT_MASK_SIZE = 14
__C.TRAIN.FG_DET_THRESH = 0.5
__C.TRAIN.FG_SEG_THRESH = 0.5
__C.TRAIN.FRACTION_SAMPLE = [0.3, 0.5, 0.2]
__C.TRAIN.THRESH_LO_SAMPLE = [0.5, 0.1, 0.0]
__C.TRAIN.THRESH_HI_SAMPLE = [1.0, 0.5, 0.1]

# Test option

__C.TEST = edict()
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3
# Set this true in the yml file to specify proposed RPN
__C.TEST.HAS_RPN = True
# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16
__C.TEST.BBOX_REG = True

# Aggregate nearby masks inside box, the box_IOU threshold
__C.TEST.MASK_MERGE_IOU_THRESH = 0.5
__C.TEST.MASK_MERGE_NMS_THRESH = 0.3
__C.TEST.CFM_INPUT_MASK_SIZE = 14

# Used for multi-scale testing, since naive implementation
# will waste a lot of on zero-padding, so we group each
# $GROUP_SCALE scales to feed in gpu. And max rois for
# each group is specified in MAX_ROIS_GPU
__C.TEST.MAX_ROIS_GPU = [2000]
__C.TEST.GROUP_SCALE = 1

# 0 means use all the MCG proposals
__C.TEST.USE_TOP_K_MCG = 0

# threshold for binarize a mask
__C.TEST.USE_MASK_MERGE = True
__C.TEST.USE_GPU_MASK_MERGE = True


def get_output_dir(imdb, net):
    """ Return the directory where experimental artifacts are placed.
        A canonical path is built using the name from an imdb and a network
        (if not None).
    """
    path = os.path.abspath(os.path.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return os.path.join(path, net.name)


def _merge_two_config(user_cfg, default_cfg):
    """ Merge user's config into default config dictionary, clobbering the
        options in b whenever they are also specified in a.
        Need to ensure the type of two val under same key are the same
        Do recursive merge when encounter hierarchical dictionary
    """
    if type(user_cfg) is not edict:
        return
    for key, val in user_cfg.iteritems():
        # Since user_cfg is a sub-file of default_cfg
        if not default_cfg.has_key(key):
            raise KeyError('{} is not a valid config key'.format(key))

        if type(default_cfg[key]) is not type(val):
            if isinstance(default_cfg[key], np.ndarray):
                val = np.array(val, dtype=default_cfg[key].dtype)
            else:
                raise ValueError(
                     'Type mismatch ({} vs. {}) '
                     'for config key: {}'.format(type(default_cfg[key]),
                                                 type(val), key))
        # Recursive merge config
        if type(val) is edict:
            try:
                _merge_two_config(user_cfg[key], default_cfg[key])
            except:
                print 'Error under config key: {}'.format(key)
                raise
        else:
            default_cfg[key] = val


def cfg_from_file(file_name):
    """ Load a config file and merge it into the default options.
    """
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_two_config(yaml_cfg, __C)
