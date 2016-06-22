# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import cv2
import scipy.io as sio

from transform.mask_transform import mask_overlap
from mnc_config import cfg


def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    Args:
        rec: recall
        prec: precision
        use_07_metric:
    Returns:
        ap: average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / (tp + fp)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_eval_sds(det_file, seg_file, devkit_path, image_list, cls_name, cache_dir,
                 class_names, ov_thresh=0.5):
    # 1. Check whether ground truth cache file exists
    with open(image_list, 'r') as f:
        lines = f.readlines()
    image_names = [x.strip() for x in lines]
    check_voc_sds_cache(cache_dir, devkit_path, image_names, class_names)
    gt_cache = cache_dir + '/' + cls_name + '_mask_gt.pkl'
    with open(gt_cache, 'rb') as f:
        gt_pkl = cPickle.load(f)

    # 2. Get predict pickle file for this class
    with open(det_file, 'rb') as f:
        boxes_pkl = cPickle.load(f)
    with open(seg_file, 'rb') as f:
        masks_pkl = cPickle.load(f)

    # 3. Pre-compute number of total instances to allocate memory
    num_image = len(image_names)
    box_num = 0
    for im_i in xrange(num_image):
        box_num += len(boxes_pkl[im_i])

    # 4. Re-organize all the predicted boxes
    new_boxes = np.zeros((box_num, 5))
    new_masks = np.zeros((box_num, cfg.MASK_SIZE, cfg.MASK_SIZE))
    new_image = []
    cnt = 0
    for image_ind in xrange(len(image_names)):
        boxes = boxes_pkl[image_ind]
        masks = masks_pkl[image_ind]
        num_instance = len(boxes)
        for box_ind in xrange(num_instance):
            new_boxes[cnt] = boxes[box_ind]
            new_masks[cnt] = masks[box_ind]
            new_image.append(image_names[image_ind])
            cnt += 1

    # 5. Rearrange boxes according to their scores
    seg_scores = new_boxes[:, -1]
    keep_inds = np.argsort(-seg_scores)
    new_boxes = new_boxes[keep_inds, :]
    new_masks = new_masks[keep_inds, :, :]
    num_pred = new_boxes.shape[0]

    # 6. Calculate t/f positive
    fp = np.zeros((num_pred, 1))
    tp = np.zeros((num_pred, 1))
    for i in xrange(num_pred):
        pred_box = np.round(new_boxes[i, :4]).astype(int)
        pred_mask = new_masks[i]
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (pred_box[2] - pred_box[0] + 1, pred_box[3] - pred_box[1] + 1))
        pred_mask = pred_mask >= cfg.BINARIZE_THRESH
        image_index = new_image[keep_inds[i]]

        if image_index not in gt_pkl:
            fp[i] = 1
            continue
        gt_dict_list = gt_pkl[image_index]
        # calculate max region overlap
        cur_overlap = -1000
        cur_overlap_ind = -1
        for ind2, gt_dict in enumerate(gt_dict_list):
            gt_mask_bound = np.round(gt_dict['mask_bound']).astype(int)
            pred_mask_bound = pred_box
            ov = mask_overlap(gt_mask_bound, pred_mask_bound, gt_dict['mask'], pred_mask)
            if ov > cur_overlap:
                cur_overlap = ov
                cur_overlap_ind = ind2
        if cur_overlap >= ov_thresh:
            if gt_dict_list[cur_overlap_ind]['already_detect']:
                fp[i] = 1
            else:
                tp[i] = 1
                gt_dict_list[cur_overlap_ind]['already_detect'] = 1
        else:
            fp[i] = 1

    # 7. Calculate precision
    num_pos = 0
    for key, val in gt_pkl.iteritems():
        num_pos += len(val)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_pos)
    # avoid divide by zero in case the first matches a difficult gt
    prec = tp / np.maximum(fp+tp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True)
    return ap


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def parse_inst(image_name, devkit_path):
    """
    Get cooresponding masks, boxes, classes according to image name
    Args:
        image_name: input image name
        devkit_path: root dir for devkit SDS
    Returns:
        roi/mask dictionary of this image
    """
    gt_mask_im_name = os.path.join(devkit_path, 'inst',
                                   image_name + '.mat')
    gt_inst_mat = sio.loadmat(gt_mask_im_name)
    gt_inst_data = gt_inst_mat['GTinst']['Segmentation'][0][0]
    gt_mask_class_name = os.path.join(devkit_path, 'cls',
                                      image_name + '.mat')
    gt_cls_mat = sio.loadmat(gt_mask_class_name)
    gt_cls_data = gt_cls_mat['GTcls']['Segmentation'][0][0]
    unique_inst = np.unique(gt_inst_data)
    # delete background pixels
    background_ind = np.where(unique_inst == 0)[0]
    unique_inst = np.delete(unique_inst, background_ind)
    record = []
    for inst_ind in xrange(unique_inst.shape[0]):
        [r, c] = np.where(gt_inst_data == unique_inst[inst_ind])
        mask_bound = np.zeros(4)
        mask_bound[0] = np.min(c)
        mask_bound[1] = np.min(r)
        mask_bound[2] = np.max(c)
        mask_bound[3] = np.max(r)
        mask = gt_inst_data[mask_bound[1]:mask_bound[3]+1, mask_bound[0]:mask_bound[2]+1]
        mask = (mask == unique_inst[inst_ind])
        mask_cls = gt_cls_data[mask_bound[1]:mask_bound[3]+1, mask_bound[0]:mask_bound[2]+1]
        mask_cls = mask_cls[mask]
        num_cls = np.unique(mask_cls)
        assert num_cls.shape[0] == 1
        cur_inst = num_cls[0]
        record.append({
            'mask': mask,
            'mask_cls': cur_inst,
            'mask_bound': mask_bound
        })

    return record


def check_voc_sds_cache(cache_dir, devkit_path, image_names, class_names):
    """
    Args:
        cache_dir: output directory for cached mask annotation
        devkit_path: root directory of VOCdevkitSDS
        image_names: used for parse image instances
        class_names: VOC 20 class names
    """

    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    exist_cache = True
    for cls_name in class_names:
        if cls_name == '__background__':
            continue
        cache_name = os.path.join(cache_dir, cls_name + '_mask_gt.pkl')
        if not os.path.isfile(cache_name):
            exist_cache = False
            break

    if not exist_cache:
        # load annotations:
        # create a list with size classes
        record_list = [{} for _ in xrange(21)]
        for i, image_name in enumerate(image_names):
            record = parse_inst(image_name, devkit_path)
            for j, mask_dic in enumerate(record):
                cls = mask_dic['mask_cls']
                mask_dic['already_detect'] = False
                if image_name not in record_list[cls]:
                    record_list[cls][image_name] = []
                record_list[cls][image_name].append(mask_dic)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(i + 1, len(image_names))

        print 'Saving cached annotations...'
        for cls_ind, name in enumerate(class_names):
            if name == '__background__':
                continue
            cachefile = os.path.join(cache_dir, name + '_mask_gt.pkl')
            with open(cachefile, 'w') as f:
                cPickle.dump(record_list[cls_ind], f)
