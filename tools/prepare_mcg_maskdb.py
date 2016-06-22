# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# System modules
import argparse
import os
import cPickle
import numpy as np
import scipy.io as sio
import cv2
from multiprocessing import Process
import time
import PIL
# User-defined module
import _init_paths
from mnc_config import cfg
from utils.cython_bbox import bbox_overlaps
from transform.mask_transform import mask_overlap, intersect_mask
from datasets.pascal_voc_seg import PascalVOCSeg


def parse_args():
    """ Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Prepare MCG roidb')
    parser.add_argument('--input', dest='input_dir',
                        help='folder contain input mcg proposals',
                        default='data/MCG-raw/', type=str)
    parser.add_argument('--output', dest='output_dir',
                        help='folder contain output roidb', required=True,
                        type=str)
    parser.add_argument('--gt_roi', dest='roidb', help='roidb',
                        default='data/cache/voc_2012_train_gt_roidb.pkl', type=str)
    parser.add_argument('--gt_mask', dest='maskdb', help='maskdb',
                        default='data/cache/voc_2012_train_gt_maskdb.pkl', type=str)
    parser.add_argument('-mask_sz', dest='mask_size',
                        help='compressed mask resolution',
                        default=21, type=int)
    parser.add_argument('--top_k', dest='top_k',
                        help='number of generated proposal',
                        default=-1, type=int)
    parser.add_argument('--db', dest='db_name',
                        help='train or validation',
                        default='train', type=str)
    parser.add_argument('--para_job', dest='para_job',
                        help='launch several process',
                        default='1', type=int)
    return parser.parse_args()


def process_roidb(file_start, file_end, db):

    for cnt in xrange(file_start, file_end):
        f = file_list[cnt]
        full_file = os.path.join(input_dir, f)
        output_cache = os.path.join(output_dir, f.split('.')[0] + '.mat')
        timer_tic = time.time()
        if os.path.exists(output_cache):
            continue
        mcg_mat = sio.loadmat(full_file)
        mcg_mask_label = mcg_mat['labels']
        mcg_superpixels = mcg_mat['superpixels']
        num_proposal = len(mcg_mask_label)
        mcg_boxes = np.zeros((num_proposal, 4))
        mcg_masks = np.zeros((num_proposal, mask_size, mask_size), dtype=np.bool)

        for ind_proposal in xrange(num_proposal):
            label = mcg_mask_label[ind_proposal][0][0]
            proposal = np.in1d(mcg_superpixels, label).reshape(mcg_superpixels.shape)
            [r, c] = np.where(proposal == 1)
            y1 = np.min(r)
            x1 = np.min(c)
            y2 = np.max(r)
            x2 = np.max(c)
            box = np.array([x1, y1, x2, y2])
            proposal = proposal[y1:y2+1, x1:x2+1]
            proposal = cv2.resize(proposal.astype(np.float), (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
            mcg_masks[ind_proposal, :, :] = proposal
            mcg_boxes[ind_proposal, :] = box

        if top_k != -1:
            mcg_boxes = mcg_boxes[:top_k, :]
            mcg_masks = mcg_masks[:top_k, :]

        if db == 'val':
            # if we prepare validation data, we only need its masks and boxes
            roidb = {
                'masks': (mcg_masks >= cfg.BINARIZE_THRESH).astype(bool),
                'boxes': mcg_boxes
            }
            sio.savemat(output_cache, roidb)
            use_time = time.time() - timer_tic
            print '%d/%d use time %f' % (cnt, len(file_list), use_time)

        else:
            # Otherwise we need to prepare other information like overlaps
            num_mcg = mcg_boxes.shape[0]
            gt_roidb = gt_roidbs[cnt]
            gt_maskdb = gt_maskdbs[cnt]
            gt_boxes = gt_roidb['boxes']
            gt_masks = gt_maskdb['gt_masks']
            gt_classes = gt_roidb['gt_classes']
            num_gt = gt_boxes.shape[0]
            num_all = num_gt + num_mcg
            # define output structure
            det_overlaps = np.zeros((num_all, 1))
            seg_overlaps = np.zeros((num_all, 1))
            seg_assignment = np.zeros((num_all, 1))
            mask_targets = np.zeros((num_all, mask_size, mask_size))
            # ------------------------------------------------------
            all_boxes = np.vstack((gt_boxes[:, :4], mcg_boxes)).astype(int)
            all_masks = np.zeros((num_all, mask_size, mask_size))
            for i in xrange(num_gt):
                all_masks[i, :, :] = (cv2.resize(gt_masks[i].astype(np.float),
                                                (mask_size, mask_size)))
            assert all_masks[num_gt:, :, :].shape == mcg_masks.shape
            all_masks[num_gt:, :, :] = mcg_masks
            # record bounding box overlaps
            cur_overlap = bbox_overlaps(all_boxes.astype(np.float), gt_boxes.astype(np.float))
            seg_assignment = cur_overlap.argmax(axis=1)
            det_overlaps = cur_overlap.max(axis=1)
            seg_assignment[det_overlaps == 0] = -1
            # record mask region overlaps
            seg_overlaps[:num_gt] = 1.0
            for i in xrange(num_gt, num_all):
                cur_mask = cv2.resize(all_masks[i, :, :].astype(np.float),
                                      (all_boxes[i, 2] - all_boxes[i, 0] + 1,
                                       all_boxes[i, 3] - all_boxes[i, 1] + 1)) >= cfg.BINARIZE_THRESH
                for mask_ind in xrange(len(gt_masks)):
                    gt_mask = gt_masks[mask_ind]
                    gt_roi = gt_roidb['boxes'][mask_ind]
                    cur_ov = mask_overlap(all_boxes[i, :], gt_roi, cur_mask, gt_mask)
                    seg_overlaps[i] = max(seg_overlaps[i], cur_ov)

            output_label = np.zeros((num_all, 1))
            for i in xrange(num_all):
                if seg_assignment[i] == -1:
                    continue
                cur_ind = seg_assignment[i]
                output_label[i] = gt_classes[seg_assignment[i]]
                mask_targets[i, :, :] = intersect_mask(all_boxes[i, :], gt_roidb['boxes'][cur_ind], gt_masks[cur_ind])

            # Some of the array need to insert a new axis to be consistent of savemat method
            roidb = {
                'masks': (all_masks >= cfg.BINARIZE_THRESH).astype(bool),
                'boxes': all_boxes,
                'det_overlap': det_overlaps[:, np.newaxis],
                'seg_overlap': seg_overlaps,
                'mask_targets': (mask_targets >= cfg.BINARIZE_THRESH).astype(bool),
                'gt_classes': gt_classes[:, np.newaxis],
                'output_label': output_label,
                'gt_assignment': seg_assignment[:, np.newaxis],
                'Flip': False
            }

            sio.savemat(output_cache, roidb)
            use_time = time.time() - timer_tic
            print '%d/%d use time %f' % (cnt, len(file_list), use_time)


def process_flip_masks(image_names, im_start, im_end):

    widths = [PIL.Image.open('data/VOCdevkitSDS/img/' + im_name + '.jpg').size[0] for im_name in image_names]
    cache_dir = output_dir
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    for index in xrange(im_start, im_end):
        output_cache = os.path.join(cache_dir, image_names[index] + '_flip.mat')
        if os.path.exists(output_cache):
            continue
        image_cache = os.path.join(cache_dir, image_names[index] + '.mat')
        orig_maskdb = sio.loadmat(image_cache)
        # Flip mask and mask regression targets
        masks = orig_maskdb['masks']
        mask_targets = orig_maskdb['mask_targets']
        mask_flip = masks[:, :, ::-1]
        mask_target_flip = mask_targets[:, :, ::-1]
        # Flip boxes
        boxes = orig_maskdb['boxes']
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = widths[index] - oldx2 - 1
        boxes[:, 2] = widths[index] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        # Other maskdb values are identical with original maskdb
        flip_maskdb = {
            'masks': (mask_flip >= cfg.BINARIZE_THRESH).astype(bool),
            'boxes': boxes,
            'det_overlap': orig_maskdb['det_overlap'],
            'seg_overlap': orig_maskdb['seg_overlap'],
            'mask_targets': (mask_target_flip >= cfg.BINARIZE_THRESH).astype(bool),
            'gt_classes': orig_maskdb['gt_classes'],
            'gt_assignment': orig_maskdb['gt_assignment'],
            'Flip': True,
            'output_label': orig_maskdb['output_label']
        }
        sio.savemat(output_cache, flip_maskdb)


if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    assert os.path.exists(input_dir), 'Path does not exist: {}'.format(input_dir)
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    mask_size = args.mask_size

    list_name = 'data/VOCdevkitSDS/train.txt' if args.db_name == 'train' else 'data/VOCdevkitSDS/val.txt'
    with open(list_name) as f:
        file_list = f.read().splitlines()

    # If we want to prepare training maskdb, first try to load gts
    if args.db_name == 'train':
        if os.path.exists(args.roidb) and os.path.exists(args.maskdb):
            with open(args.roidb, 'rb') as f:
                gt_roidbs = cPickle.load(f)
            with open(args.maskdb, 'rb') as f:
                gt_maskdbs = cPickle.load(f)
        else:
            db = PascalVOCSeg('train', '2012', 'data/VOCdevkitSDS/')
            gt_roidbs = db.gt_roidb()
            gt_maskdbs = db.gt_maskdb()

    top_k = args.top_k
    num_process = args.para_job
    # Prepare train/val maskdb use multi-process
    processes = []
    file_start = 0
    file_offset = int(np.ceil(len(file_list) / float(num_process)))
    for process_id in xrange(num_process):
        file_end = min(file_start + file_offset, len(file_list))
        p = Process(target=process_roidb, args=(file_start, file_end, args.db_name))
        p.start()
        processes.append(p)
        file_start += file_offset

    for p in processes:
        p.join()

    # If db_name == 'train', we still need to add flipped maskdb into output folder
    # Add flipped mask and mask regression targets after prepare the original mcg proposals
    if args.db_name == 'train':
        print 'Appending flipped MCG to ROI'
        processes = []
        file_start = 0
        file_offset = int(np.ceil(len(file_list) / float(num_process)))
        for process_id in xrange(num_process):
            file_end = min(file_start + file_offset, len(file_list))
            p = Process(target=process_flip_masks, args=(file_list, file_start, file_end))
            p.start()
            processes.append(p)
            file_start += file_offset
        for p in processes:
            p.join()
