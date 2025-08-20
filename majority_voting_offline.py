import sys
import os
import numpy as np
import time
import math
import torch
from pruning import iou_2d
from lookup_table import color_list


def rotate_point(x, y, yaw):
    """Rotate a point (x, y) by a yaw angle around the origin (0, 0)."""
    yaw_rad = math.radians(yaw)
    x_rot = x * math.cos(yaw_rad) - y * math.sin(yaw_rad)
    y_rot = x * math.sin(yaw_rad) + y * math.cos(yaw_rad)
    return x_rot, y_rot

def rotate_bounding_box(x, y, dx, dy, yaw):
    """Rotate the corners of a bounding box and return the rotated min/max corners."""
    half_dx = dx / 2
    half_dy = dy / 2
    
    corners = [
        (x - half_dx, y - half_dy),
        (x + half_dx, y - half_dy),
        (x + half_dx, y + half_dy),
        (x - half_dx, y + half_dy),
    ]
    
    rotated_corners = [rotate_point(cx, cy, yaw) for cx, cy in corners]
    
    rotated_x_min = min(c[0] for c in rotated_corners)
    rotated_x_max = max(c[0] for c in rotated_corners)
    rotated_y_min = min(c[1] for c in rotated_corners)
    rotated_y_max = max(c[1] for c in rotated_corners)
    
    return rotated_x_min, rotated_x_max, rotated_y_min, rotated_y_max


def majority_voting(bboxes_RGBD, pred_RGBD, conf_RGBD, bboxes_LIDAR, pred_LIDAR, conf_LIDAR):
    majo_pred_list = []
    pred_RGBD = pred_RGBD.reshape(-1, 1) 
    pred_LIDAR = pred_LIDAR.reshape(-1, 1) 
    ignore = []
    confs = []
    counter = 0
    
    for i, rgbd in enumerate(bboxes_RGBD):
        overlap = False
        for j, lidar in enumerate(bboxes_LIDAR):
            if iou_2d(rgbd, lidar) > 0.5 and j not in ignore:
                counter += 1
                overlap = True
                break
        if overlap:
            if conf_RGBD[i] > conf_LIDAR[j]:
                ignore.append(j)
                confs.append(conf_RGBD[i])
                majo_pred_list.append(np.concatenate((rgbd, pred_RGBD[i]), axis=0))
            else:
                confs.append(conf_LIDAR[j])
                majo_pred_list.append(np.concatenate((lidar, pred_LIDAR[j]), axis=0))
        else:
            confs.append(conf_RGBD[i])
            majo_pred_list.append(np.concatenate((rgbd, pred_RGBD[i]), axis=0))

    for j, lidar in enumerate(bboxes_LIDAR):
        if j not in ignore:
            # Append LIDAR predictions if not already ignored
            confs.append(conf_LIDAR[j])
            majo_pred_list.append(np.concatenate((lidar, pred_LIDAR[j]), axis=0))

    majo_pred = np.array(majo_pred_list)
    return majo_pred, np.array(confs), counter
