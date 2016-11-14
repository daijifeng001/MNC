#!/usr/bin/python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import os
import argparse
import time
import cv2
import numpy as np
# User-defined module
import _init_paths
import caffe
from mnc_config import cfg
from transform.bbox_transform import clip_boxes
from utils.blob import prep_im_for_blob, im_list_to_blob
from transform.mask_transform import gpu_mask_voting
import matplotlib.pyplot as plt
from utils.vis_seg import _convert_pred_to_image, _get_voc_color_map
from PIL import Image

# VOC 20 classes
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='MNC demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='./models/VGG16/mnc_5stage/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='./data/mnc_model/mnc_model.caffemodel.h5', type=str)

    args = parser.parse_args()
    return args


def prepare_mnc_args(im, net):
    # Prepare image data blob
    blobs = {'data': None}
    processed_ims = []
    im, im_scale_factors = \
        prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
    processed_ims.append(im)
    blobs['data'] = im_list_to_blob(processed_ims)
    # Prepare image info blob
    im_scales = [np.array(im_scale_factors)]
    assert len(im_scales) == 1, 'Only single-image batch implemented'
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)
    # Reshape network inputs and do forward
    net.blobs['data'].reshape(*blobs['data'].shape)
    net.blobs['im_info'].reshape(*blobs['im_info'].shape)
    forward_kwargs = {
        'data': blobs['data'].astype(np.float32, copy=False),
        'im_info': blobs['im_info'].astype(np.float32, copy=False)
    }
    return forward_kwargs, im_scales


def im_detect(im, net):
    forward_kwargs, im_scales = prepare_mnc_args(im, net)
    blobs_out = net.forward(**forward_kwargs)
    # output we need to collect:
    # 1. output from phase1'
    rois_phase1 = net.blobs['rois'].data.copy()
    masks_phase1 = net.blobs['mask_proposal'].data[...]
    scores_phase1 = net.blobs['seg_cls_prob'].data[...]
    # 2. output from phase2
    rois_phase2 = net.blobs['rois_ext'].data[...]
    masks_phase2 = net.blobs['mask_proposal_ext'].data[...]
    scores_phase2 = net.blobs['seg_cls_prob_ext'].data[...]
    # Boxes are in resized space, we un-scale them back
    rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
    rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
    rois_phase1, _ = clip_boxes(rois_phase1, im.shape)
    rois_phase2, _ = clip_boxes(rois_phase2, im.shape)
    # concatenate two stages to get final network output
    masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
    boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
    scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
    return boxes, masks, scores


def get_vis_dict(result_box, result_mask, img_name, cls_names, vis_thresh=0.5):
    box_for_img = []
    mask_for_img = []
    cls_for_img = []
    for cls_ind, cls_name in enumerate(cls_names):
        det_for_img = result_box[cls_ind]
        seg_for_img = result_mask[cls_ind]
        keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
        for keep in keep_inds:
            box_for_img.append(det_for_img[keep])
            mask_for_img.append(seg_for_img[keep][0])
            cls_for_img.append(cls_ind + 1)
    res_dict = {'image_name': img_name,
                'cls_name': cls_for_img,
                'boxes': box_for_img,
                'masks': mask_for_img}
    return res_dict

if __name__ == '__main__':
    args = parse_args()
    test_prototxt = args.prototxt
    test_model = args.caffemodel

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    net = caffe.Net(test_prototxt, test_model, caffe.TEST)

    # Warm up for the first two images
    im = 128 * np.ones((300, 500, 3), dtype=np.float32)
    for i in xrange(2):
        _, _, _ = im_detect(im, net)

    im_names = ['2008_000533.jpg', '2008_000910.jpg', '2008_001602.jpg',
                '2008_001717.jpg', '2008_008093.jpg']
    demo_dir = './data/demo'
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        gt_image = os.path.join(demo_dir, im_name)
        im = cv2.imread(gt_image)
        start = time.time()
        boxes, masks, seg_scores = im_detect(im, net)
        end = time.time()
        print 'forward time %f' % (end-start)
        result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, len(CLASSES) + 1,
                                                  100, im.shape[1], im.shape[0])
        pred_dict = get_vis_dict(result_box, result_mask, 'data/demo/' + im_name, CLASSES)

        img_width = im.shape[1]
        img_height = im.shape[0]
        
        inst_img, cls_img = _convert_pred_to_image(img_width, img_height, pred_dict)
        color_map = _get_voc_color_map()
        target_cls_file = os.path.join(demo_dir, 'cls_' + im_name)
        cls_out_img = np.zeros((img_height, img_width, 3))
        for i in xrange(img_height):
            for j in xrange(img_width):
                cls_out_img[i][j] = color_map[cls_img[i][j]][::-1]
        cv2.imwrite(target_cls_file, cls_out_img)
        
        background = Image.open(gt_image)
        mask = Image.open(target_cls_file)
        background = background.convert('RGBA')
        mask = mask.convert('RGBA')
        superimpose_image = Image.blend(background, mask, 0.8)
        superimpose_name = os.path.join(demo_dir, 'final_' + im_name)
        superimpose_image.save(superimpose_name, 'JPEG')
        im = cv2.imread(superimpose_name)

        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        classes = pred_dict['cls_name']
        for i in xrange(len(classes)):
            score = pred_dict['boxes'][i][-1]
            bbox = pred_dict['boxes'][i][:4]
            cls_ind = classes[i] - 1
            ax.text(bbox[0], bbox[1] - 8,
                '{:s} {:.4f}'.format(CLASSES[cls_ind], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.draw()

        fig.savefig(os.path.join(demo_dir, im_name[:-4]+'.png'))
        os.remove(superimpose_name)
        os.remove(target_cls_file)
