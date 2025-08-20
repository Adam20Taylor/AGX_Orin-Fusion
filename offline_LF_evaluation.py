import sys
import os
import numpy as np
import time
import struct
import math
import random
import json
from load_JSON import get_bboxes
import lookup_table
from majority_voting_offline import majority_voting
from iou_3d import compute_recall_precisions, average_precision, get_TruePositive, get_nr_gt
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from offline_publish_gt import publish_bounding_boxes, publish_points, corner_talker, publish_colored_points, get_camera_info
from knn import do_knn
import rosbag
from pruning import prune

from ros_numpy import point_cloud2
sys.path.insert(1,'/home/jetson/theadams2/mmdetection3d')
from mmdetection3d.mmdet3d.apis import init_model, inference_detector


config_path_LIDAR = '/home/jetson/theadams2/mmdetection3d/configs/votenet/votenet_head-iouloss_8xb8_scannet-3d.py'
checkpoint_path_LIDAR = '/home/jetson/theadams2/weights/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth'
config_path_RGBD = '/home/jetson/theadams2/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py'
checkpoint_path_RGBD = '/home/jetson/theadams2/weights/tr3d_1xb16_scannet-3d-18class.pth'

def talker(frames):
    # Input: list of frames to publish to ros topic
    i = 0
    rospy.init_node('votenet_test', anonymous=True)
    pub_IM = rospy.Publisher('/detection/Camera_image', Image, queue_size=1)
    pub_LP = rospy.Publisher('/detection/Lidar_predictions', MarkerArray, queue_size=1)
    pub_GT = rospy.Publisher('/detection/Ground_Truth', MarkerArray, queue_size=1)
    pub_PR = rospy.Publisher('/detection/Predictions', MarkerArray, queue_size=1)
    pub_PC = rospy.Publisher('/detection/points', PointCloud2, queue_size=1)
    pub_CI = rospy.Publisher('/detection/camera_info', CameraInfo, queue_size=1)

    image_list = []
    with rosbag.Bag('/home/jetson/theadams2/outbag1.bag', 'r') as bag:
        for _,msg,t in bag.read_messages(topics='/robot/front_rgbd_camera/color/image_raw'):
            image_list.append((msg, t))

    rate = rospy.Rate(0.5) # Hz
    ids = frames
    while not rospy.is_shutdown():
        points = publish_colored_points(files_RGBD[ids[i]])
        if len(preds[ids[i]]) >0:
            markers = publish_bounding_boxes(gt_bboxes_list[ids[i]], None)
            pred_frame = publish_bounding_boxes(preds[ids[i]][:,0:7], preds[ids[i]][:,7].astype(int))
            pub_PC.publish(points)
            pub_GT.publish(markers)
            pub_PR.publish(pred_frame)
        pub_IM.publish(image_list[ids[i]][0])
        cam_info = get_camera_info()
        cam_info.header.stamp = image_list[ids[i]][1]
        pub_CI.publish(cam_info)

        print(f"Publishing: {ids[i]}")
        if i + 1 == len(ids):
            i = 0
        else:
            i += 1
        rate.sleep()


num_boxes = 20
gt_file = "/home/jetson/theadams2/Cropped_LIDAR_Dataset-v0.4.json"
RGBD_dir = "/home/jetson/theadams2/dataset/sequences/_robot_front_rgbd_camera_depth_color_points"
LIDAR_dir = "/home/jetson/theadams2/unique_dataset/sequences/_robot_top_3d_laser_points"

useLIDAR = True     # Use VoteNet, if both useRGBD and useLIDAR -> uses late fusion
useRGBD = True      # Use TR3D, if both useRGBD and useLIDAR -> uses late fusion
useEarly = True     # Use Early fusion
threshold = 0.5     # Confidence/probability threshold
iou = 0.25          # IoU threshold
nr_runs = 5         # number of runs


files_LIDAR = sorted([f"{LIDAR_dir}/{f}" for f in os.listdir(LIDAR_dir) if f.endswith('.bin')])
files_RGBD = sorted([f"{RGBD_dir}/{f}" for f in os.listdir(RGBD_dir) if f.endswith('.bin')])
with open(gt_file) as f:
    data = json.load(f)

frames = data['dataset']['samples'][0]['labels']['ground-truth']['attributes']['frames']
gt_bboxes_list = get_bboxes(frames)


if useRGBD:
    print("Running TR3D")
    RGBD_model = init_model(config_path_RGBD, checkpoint_path_RGBD, device='cuda:0')

if useLIDAR:
    print("Running Votenet")
    LIDAR_model = init_model(config_path_LIDAR, checkpoint_path_LIDAR, device='cuda:0')

mean_times = []
mean_times_wo_first = []
maps = []


for j in range(nr_runs):
    preds = []
    times = []
    
    
    for i, frame in enumerate(files_LIDAR):
        print(f"Frame number {i, j}")
        start = time.time()
        if useLIDAR:
            LIDAR_results = inference_detector(LIDAR_model, frame)
            LIDAR_bboxes_pred = LIDAR_results[0].pred_instances_3d.bboxes_3d.cpu().numpy()
            LIDAR_scores = LIDAR_results[0].pred_instances_3d.scores_3d.cpu().numpy()
            LIDAR_labels_pred = LIDAR_results[0].pred_instances_3d.labels_3d.cpu().numpy()
        if useRGBD:
            if useEarly:
                ef_frame = do_knn(files_RGBD[i],files_LIDAR[i])
                RGBD_results = inference_detector(RGBD_model, ef_frame)
                RGBD_bboxes_pred = RGBD_results[0][0].pred_instances_3d.bboxes_3d.cpu().numpy()
                RGBD_scores = RGBD_results[0][0].pred_instances_3d.scores_3d.cpu().numpy()
                RGBD_labels_pred = RGBD_results[0][0].pred_instances_3d.labels_3d.cpu().numpy()
            else: 
                RGBD_results = inference_detector(RGBD_model, files_RGBD[i])
                RGBD_bboxes_pred = RGBD_results[0].pred_instances_3d.bboxes_3d.cpu().numpy()
                RGBD_scores = RGBD_results[0].pred_instances_3d.scores_3d.cpu().numpy()
                RGBD_labels_pred = RGBD_results[0].pred_instances_3d.labels_3d.cpu().numpy()
        if useLIDAR and useRGBD:
            pred, scores,_ = majority_voting(RGBD_bboxes_pred[RGBD_scores > threshold], RGBD_labels_pred[RGBD_scores > threshold],
                                RGBD_scores[RGBD_scores > threshold], LIDAR_bboxes_pred[LIDAR_scores > threshold],
                                LIDAR_labels_pred[LIDAR_scores > threshold], LIDAR_scores[LIDAR_scores > threshold])
            score = scores[scores > threshold]
            
        elif useLIDAR:
            pred = np.concatenate((LIDAR_bboxes_pred[LIDAR_scores > threshold], 
                                LIDAR_labels_pred[LIDAR_scores > threshold].reshape(-1,1)), axis=1)
            score = LIDAR_scores[LIDAR_scores > threshold]
        elif useRGBD:
            pred = np.concatenate((RGBD_bboxes_pred[RGBD_scores > threshold], 
                                RGBD_labels_pred[RGBD_scores > threshold].reshape(-1,1)), axis=1)
            score = RGBD_scores[RGBD_scores > threshold]
        pred, score = prune(pred, score)
        times.append(time.time()-start)
        preds.append(pred)
        chair_precisions = []
        table_precisions = []
        bookshelf_precisions = []
        
        get_TruePositive(gt_bboxes_list[i], pred, score, i, "mmdet", iou)

        mean_times.append(np.mean(times))
        if len(times) > 1:
            
            mean_times_wo_first.append(np.mean(times[1:]))
nr_gt = get_nr_gt(gt_bboxes_list)
recall_chair, precision_chair = compute_recall_precisions("chair", nr_gt["chair"]*nr_runs)
ap_chair = average_precision(recall_chair, precision_chair, mode="area")
recall_table, precision_table = compute_recall_precisions("table", nr_gt["table"]*nr_runs)
ap_table = average_precision(recall_table, precision_table, mode="area")
recall_bookshelf, precision_bookshelf = compute_recall_precisions("bookshelf", nr_gt["bookshelf"]*nr_runs)
ap_bookshelf = average_precision(recall_bookshelf, precision_bookshelf, mode="area")

mAP = np.mean([ap_chair, ap_table, ap_bookshelf])

print(f"Mean time:                   {np.round(np.mean(mean_times), 3)}")
print(f"Mean time without first:     {np.round(np.mean(mean_times_wo_first), 3)}")
print("---------------------------------------------")
print(f"AP {iou} chairs:              {np.round(ap_chair,3)}")
print(f"AP {iou} table:               {np.round(ap_table,3)}")
print(f"AP {iou} bookshelf:           {np.round(ap_bookshelf,3)}")
print(f"mAP {iou}:                    {np.round(mAP,3)}")
print("---------------------------------------------")
