import numpy as np
from collections import defaultdict
import json
import sys
import load_JSON
from lookup_table import pred_classes
from lookup_table import gt_classes
import torch
from pytorch3d.ops import box3d_overlap
import time

corners = []

def create_tensor(box, pred_mode):
    """
    Creates torch tensors of box corners used for box3d_overlap function.
    """
    box_array = np.zeros((1,8,3))
    offsets = np.array([[-1,1,-1],[1,1,-1], [1,-1,-1], [-1,-1,-1],[-1,1,1], [1,1,1], [1,-1,1], [-1,-1,1]])
    if type(box) == dict:
        t = box['yaw']
        R1 = np.array([[np.cos(t), -np.sin(t)], 
                       [np.sin(t), np.cos(t)]])
        half_lens = np.array([box['dx']/2, box['dy']/2, box['dz']/2])
        centers = np.array([box['x'], box['y'], box['z']])
    else:
        half_lens = np.array([box[3]/2, box[4]/2, box[5]/2])
        if pred_mode == "mmdet":
            centers = np.array([box[0], box[1], box[2]+half_lens[2]]) # For mmdet models
        else:
            centers = np.array([box[0], box[1], box[2]])  #For torch models
    corners = (half_lens*offsets)
    if type(box) == dict:
        corners[:,0:2] = centers[0:2] + (R1 @ corners[:,0:2].T).T 
        corners[:,2] = centers[2] + corners[:,2]
        box_array[0,:,:] = corners
    else:
        box_array[0,:,:] = centers + (half_lens*offsets)
    return_array = np.zeros((8,3))
    return_array[:,:] = np.copy(box_array[0,:,:])
    return torch.tensor(box_array, dtype=torch.float32), return_array
        

def iou_3d(box1, box2, p_class):
    vol, iou = box3d_overlap(box1,box2)
    return iou

def get_nr_gt(gts):
    nr_gt = {
    "chair": 0,
    "table": 0,
    "bookshelf": 0
    }
    for frame in gts:
        for gt in frame:
            if gt["class_id"] == 1:
                nr_gt["chair"] += 1
            elif gt["class_id"] == 3:
                nr_gt["table"] += 1
            elif gt["class_id"] == 2:
                nr_gt["bookshelf"] += 1
    return nr_gt

tp_flags = {
    "chair": [],
    "table": [],
    "bookshelf": []
}
confs = {
    "chair": [],
    "table": [],
    "bookshelf": []
}

def get_TruePositive(gt_boxes, pred_boxes, confindences,index, pred_mode, iou_threshold=0.25):
    global corners
    global tp_flags
    global confs
    """
    Evaluate 3D object detection models with class labels.
    Each ground truth and predicted box should have (x, y, z, w, h, l, θ, class).
    """
    matched_gt = []
    for i, p_box in enumerate(pred_boxes):
        p_class = pred_classes[p_box[-1]]  # Extract class label
        if p_class == "cabinet":            # combine cabinet and bookshelf classes
            p_class = "bookshelf"
        ious = []
        for g_box in gt_boxes:
            gt_tensor, gt_corners = create_tensor(g_box, pred_mode)
            p_tensor, p_corners = create_tensor(p_box, pred_mode)
            iou = (iou_3d(gt_tensor, p_tensor, p_class), g_box)
            if iou[0][0] >= 0:
                corners.append([g_box,p_box,gt_corners,p_corners,iou])
                ious.append(iou)

        ious = sorted(ious, key=lambda x: x[0], reverse=True)
        if ious[0][0] > iou_threshold:
            best_iou, best_gt = ious[0]
            gt_class = gt_classes[best_gt['class_id']]
            best = [best_gt['x'], best_gt['y'],best_gt['z']]
            if best not in matched_gt:  # Avoid multiple matches to the same GT
                if gt_class == p_class:
                    tp_flags[f"{p_class}"].append(1)
                    confs[f"{p_class}"].append(confindences[i])
                else:
                    if p_class in gt_classes:
                        tp_flags[f"{p_class}"].append(0) # Wrong classification
                        confs[f"{p_class}"].append(confindences[i])
                matched_gt.append(best)
        else:
            if p_class in gt_classes:
                tp_flags[f"{p_class}"].append(0) # False positive (low IoU)
                confs[f"{p_class}"].append(confindences[i])


def average_precision(recalls, precisions, mode='area'):
    """Source: MMdet3D
    Calculate average precision (for single or multiple scales). 

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets) \
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of \
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    
    return ap

def compute_recall_precisions(class_name,num_gt):
    global confs
    global tp_flags
    sorted_indices = np.argsort(-np.array(confs[f"{class_name}"]))
    sorted_tp_flags = np.array(tp_flags[f"{class_name}"])[sorted_indices]

    tp_cum = np.cumsum(sorted_tp_flags)
    sorted_fp_flags = [not flag for flag in sorted_tp_flags]
    fp_cum = np.cumsum(sorted_fp_flags)

    recalls = tp_cum / num_gt
    precisions = tp_cum / (tp_cum + fp_cum)
    tp_flags[f"{class_name}"] = []
    confs[f"{class_name}"] = []
    return recalls, precisions

def compute_ap(class_name,num_gt):
    # Sort detections by confidence
    global confs
    global tp_flags
    sorted_indices = np.argsort(-np.array(confs[f"{class_name}"]))
    sorted_tp_flags = np.array(tp_flags[f"{class_name}"])[sorted_indices]

    tp_cum = np.cumsum(sorted_tp_flags)
    sorted_fp_flags = [not flag for flag in sorted_tp_flags]
    fp_cum = np.cumsum(sorted_fp_flags)

    recalls = tp_cum / num_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return ap
    