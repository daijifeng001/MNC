# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import cPickle
import os
import cv2
from PIL import Image
from mnc_config import cfg


def vis_seg(img_names, cls_names, output_dir, gt_dir):
    """
    This function plot segmentation results to specific directory
    Args:
        img_names: list
    """
    assert os.path.exists(output_dir)
    # a list of dictionary
    inst_dir = os.path.join(output_dir, 'SegInst')
    cls_dir = os.path.join(output_dir, 'SegCls')
    res_dir = os.path.join(output_dir, 'SegRes')
    if not os.path.isdir(inst_dir):
        os.mkdir(inst_dir)
    if not os.path.isdir(cls_dir):
        os.mkdir(cls_dir)
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    res_list = _prepare_dict(img_names, cls_names, output_dir)
    for img_ind, image_name in enumerate(img_names):
        target_inst_file = os.path.join(inst_dir, image_name + '.jpg')
        target_cls_file = os.path.join(cls_dir, image_name + '.jpg')
        print image_name
        gt_image = gt_dir + '/img/' + image_name + '.jpg'
        img_data = cv2.imread(gt_image)
        img_width = img_data.shape[1]
        img_height = img_data.shape[0]
        pred_dict = res_list[img_ind]
        inst_img, cls_img = _convert_pred_to_image(img_width, img_height, pred_dict)
        color_map = _get_voc_color_map()
        inst_out_img = np.zeros((img_height, img_width, 3))
        cls_out_img = np.zeros((img_height, img_width, 3))
        for i in xrange(img_height):
            for j in xrange(img_width):
                inst_out_img[i][j] = color_map[inst_img[i][j]][::-1]
                cls_out_img[i][j] = color_map[cls_img[i][j]][::-1]

        cv2.imwrite(target_inst_file, inst_out_img)
        cv2.imwrite(target_cls_file, cls_out_img)
        background = Image.open(gt_image)
        mask = Image.open(target_cls_file)
        background = background.convert('RGBA')
        mask = mask.convert('RGBA')
        superimpose_image = Image.blend(background, mask, 0.8)
        name = os.path.join(res_dir, image_name + '.png')
        superimpose_image.save(name, 'PNG')


def _prepare_dict(img_names, cls_names, cache_dir, vis_thresh=0.5):
    """
    Returns:
        list, each list is a dictionary contains mask list, box list
    """
    res_list = []
    det_file = os.path.join(cache_dir, 'res_boxes.pkl')
    with open(det_file, 'rb') as f:
        det_pkl = cPickle.load(f)
    seg_file = os.path.join(cache_dir, 'res_masks.pkl')
    with open(seg_file, 'rb') as f:
        seg_pkl = cPickle.load(f)

    for img_ind, image_name in enumerate(img_names):
        box_for_img = []
        mask_for_img = []
        cls_for_img = []
        for cls_ind, cls_name in enumerate(cls_names):
            if cls_name == '__background__' or len(det_pkl[cls_ind][img_ind]) == 0:
                continue
            det_for_img = det_pkl[cls_ind][img_ind]
            seg_for_img = seg_pkl[cls_ind][img_ind]
            keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
            for keep in keep_inds:
                box_for_img.append(det_for_img[keep])
                # TODO: remove this annoying 0
                mask_for_img.append(seg_for_img[keep][0])
                cls_for_img.append(cls_ind)
        res_dict = {'image_name': image_name,
                    'cls_name': cls_for_img,
                    'boxes': box_for_img,
                    'masks': mask_for_img}
        res_list.append(res_dict)

    return res_list


def _convert_pred_to_image(img_width, img_height, pred_dict):
    num_inst = len(pred_dict['boxes'])
    inst_img = np.zeros((img_height, img_width))
    cls_img = np.zeros((img_height, img_width))
    for i in xrange(num_inst):
        box = np.round(pred_dict['boxes'][i]).astype(int)
        mask = pred_dict['masks'][i]
        cls_num = pred_dict['cls_name'][i]
        # clip box into image space
        box[0] = min(max(box[0], 0), img_width - 1)
        box[1] = min(max(box[1], 0), img_height - 1)
        box[2] = min(max(box[2], 0), img_width - 1)
        box[3] = min(max(box[3], 0), img_height - 1)
        mask = cv2.resize(mask.astype(np.float32), (box[2]-box[0]+1, box[3]-box[1]+1))
        mask = mask >= cfg.BINARIZE_THRESH

        part1 = (i+1) * mask.astype(np.float32)
        part2 = np.multiply(np.logical_not(mask), inst_img[box[1]:box[3]+1, box[0]:box[2]+1])
        part3 = np.multiply(np.logical_not(mask), cls_img[box[1]:box[3]+1, box[0]:box[2]+1])
        inst_img[box[1]:box[3]+1, box[0]:box[2]+1] = part1 + part2
        cls_img[box[1]:box[3]+1, box[0]:box[2]+1] = cls_num * mask.astype(np.float32) + part3
        # Plot bounding boxes simultaneously
        cls_img[box[1]:box[3]+1, box[0]-1:box[0]+1] = 150
        cls_img[box[1]:box[3]+1, box[2]-1:box[2]+1] = 150
        cls_img[box[1]-1:box[1]+1, box[0]:box[2]+1] = 150
        cls_img[box[3]-1:box[3]+1, box[0]:box[2]+1] = 150

    inst_img = inst_img.astype(int)
    cls_img = cls_img.astype(int)
    return inst_img, cls_img


def _get_voc_color_map(n=256):
    color_map = np.zeros((n, 3))
    for i in xrange(n):
        r = b = g = 0
        cid = i
        for j in xrange(0, 8):
            r = np.bitwise_or(r, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-1], 7-j))
            g = np.bitwise_or(g, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-2], 7-j))
            b = np.bitwise_or(b, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-3], 7-j))
            cid = np.right_shift(cid, 3)

        color_map[i][0] = r
        color_map[i][1] = g
        color_map[i][2] = b
    return color_map
