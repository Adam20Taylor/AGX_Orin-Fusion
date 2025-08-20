import sys
import os
import rospy
import rosbag
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from offline_publish_gt import publish_bounding_boxes, publish_points, corner_talker, publish_colored_points, get_camera_info
import numpy as np
import time
import struct
import math
import random
import json
from load_JSON import get_bboxes
import lookup_table
from iou_3d import compute_recall_precisions, average_precision, get_TruePositive, get_nr_gt
from knn import do_knn
from majority_voting_offline import majority_voting
from pruning import prune









from MF_model import RGBDBackbone, LiDARBackbone, CombinedModel, ConcatModel, normalize_points_color
import torch
import MinkowskiEngine as ME

sys.path.insert(1,'/home/jetson/theadams2/mmdetection3d')
from mmdet3d.apis import init_model, inference_detector




def talker(frames):
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

    rate = rospy.Rate(0.1)
    ids = frames
    while not rospy.is_shutdown():
        if True:
            points = publish_points(files_LIDAR[ids[i]])
        
        
        if len(preds[ids[i]]) >0:
            if False:
                lidar_markers = publish_bounding_boxes(lidar_preds[ids[i]][:,0:7], lidar_preds[ids[i]][:,7].astype(int))
                pub_LP.publish(lidar_markers)
            markers = publish_bounding_boxes(gt_bboxes_list[ids[i]], None)
            pred_frame = publish_bounding_boxes(preds[ids[i]][:,0:-1], preds[ids[i]][:,-1].astype(int))
            pub_PC.publish(points)
            pub_PR.publish(pred_frame)
            pub_GT.publish(markers)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

files_LIDAR = sorted([f"{LIDAR_dir}/{f}" for f in os.listdir(LIDAR_dir) if f.endswith('.bin')])
files_RGBD = sorted([f"{RGBD_dir}/{f}" for f in os.listdir(RGBD_dir) if f.endswith('.bin')])
with open(gt_file) as f:
    data = json.load(f)

frames = data['dataset']['samples'][0]['labels']['ground-truth']['attributes']['frames']
gt_bboxes_list = get_bboxes(frames)



mean_sizes=np.array([[0.76966727, 0.8116021, 0.92573744],
                            [1.876858, 1.8425595, 1.1931566],
                            [0.61328, 0.6148609, 0.7182701],
                            [1.3955007, 1.5121545, 0.83443564],
                            [0.97949594, 1.0675149, 0.6329687],
                            [0.531663, 0.5955577, 1.7500148],
                            [0.9624706, 0.72462326, 1.1481868],
                            [0.83221924, 1.0490936, 1.6875663],
                            [0.21132214, 0.4206159, 0.5372846],
                            [1.4440073, 1.8970833, 0.26985747],
                            [1.0294262, 1.4040797, 0.87554324],
                            [1.3766412, 0.65521795, 1.6813129],
                            [0.6650819, 0.71111923, 1.298853],
                            [0.41999173, 0.37906948, 1.7513971],
                            [0.59359556, 0.5912492, 0.73919016],
                            [0.50867593, 0.50656086, 0.30136237],
                            [1.1511526, 1.0546296, 0.49706793],
                            [0.47535285, 0.49249494, 0.5802117]])

voxel_size = 0.02
print(device)
votenet = LiDARBackbone(num_class=18, num_heading_bin=1, num_size_cluster=18, mean_size_arr = mean_sizes).to(device)
TR3D = RGBDBackbone().to(device)

config_path = '/home/jetson/theadams2/mmdetection3d/configs/votenet/votenet_head-iouloss_8xb8_scannet-3d.py'
checkpoint_path = '/home/jetson/theadams2/weights/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth'

useSimple = True    # Use Simple concatenation instead of gating fusion
useLIDAR = True     # Use late fusion with VoteNet
useEarly = True     # Use Early fusion
threshold = 0.5     # Confidence/probability threshold
iou = 0.25          # IoU threshold
nr_runs = 5         # number of runs

if useSimple is False:
    fusion_model = ConcatModel(TR3D, votenet, voxel_size).to(device)
    state_dict = torch.load("/home/jetson/theadams2/weights/best_model_sc_20e.pth", 
                            map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                            weights_only=False)
else:
    fusion_model = CombinedModel(TR3D, votenet, voxel_size).to(device)
    state_dict = torch.load("/home/jetson/theadams2/weights/best_model_20e.pth", 
                            map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                            weights_only=False)
if useLIDAR:
    print("Running Votenet")
    LIDAR_model = init_model(config_path, checkpoint_path, device='cuda:0')

fusion_model.load_state_dict(state_dict, strict=True)
fusion_model.eval()

mean_times = []
mean_times_wo_first = []
avarage_time = []

for j in range(nr_runs):
    preds = []
    times = []
    lidar_preds = []
    
    for i, frame in enumerate(zip(files_LIDAR, files_RGBD)):
        if useEarly:
            ef_frame = do_knn(files_RGBD[i],files_LIDAR[i])
            array_camera = ef_frame[0]
        else:
            frame_C = frame[1]
            array_camera = np.fromfile(frame_C, dtype=np.float32).reshape([-1,6])
        camera_normed = normalize_points_color(array_camera)
        tensor_camera = torch.from_numpy(camera_normed)
        tensor_camera[...,0:3] = torch.floor(tensor_camera[...,0:3] / voxel_size).int()

        coords, feats = ME.utils.sparse_collate([tensor_camera[...,0:3]], [tensor_camera[...,3:6]])
        in_field = ME.TensorField(
                            features = feats,
                            coordinates = coords,
                            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                            device=device
        )
        rgb_input = in_field.sparse()

        frame_L = frame[0]
        tensor_lidar = torch.tensor(np.fromfile(frame_L, dtype=np.float32), device=device)
        tensor_lidar = tensor_lidar.view([-1,6]).unsqueeze(0)[..., 0:4]
        votenet_input = {"point_clouds": tensor_lidar}

        print(f"Frame number {i, j}")
        start = time.time()
        
        with torch.no_grad():
            bb, cl, po, scores = fusion_model(rgb_input, votenet_input)
        bb[..., 0:3] = bb[..., 0:3] + po
        zeros_to_add = np.zeros((bb[scores > threshold].shape[0],1))
        pred = np.concatenate((bb[scores > threshold].cpu().numpy(),zeros_to_add, 
                                torch.argmax(cl[scores > threshold], dim=1).cpu().numpy().reshape(-1,1)), axis=1)
        score = scores[scores > threshold].cpu().numpy()
        pred, score = prune(pred, score)
        if useLIDAR:
            LIDAR_results = inference_detector(LIDAR_model, frame[0])
            LIDAR_bboxes_pred = LIDAR_results[0].pred_instances_3d.bboxes_3d.cpu().numpy()
            
            LIDAR_bboxes_pred[:,2] = LIDAR_bboxes_pred[:,2] + LIDAR_bboxes_pred[:,5]/2
            
            LIDAR_scores = LIDAR_results[0].pred_instances_3d.scores_3d.cpu().numpy()
            LIDAR_labels_pred = LIDAR_results[0].pred_instances_3d.labels_3d.cpu().numpy()
            if len(pred) == 0:
                pred, scores,_ = majority_voting(np.array([]), np.array([]), np.array([]), 
                                    LIDAR_bboxes_pred[LIDAR_scores > threshold],
                                    LIDAR_labels_pred[LIDAR_scores > threshold], LIDAR_scores[LIDAR_scores > threshold])
            else:
                pred, scores,_ = majority_voting(pred[:,0:7], pred[:,7], score, 
                                    LIDAR_bboxes_pred[LIDAR_scores > threshold],
                                    LIDAR_labels_pred[LIDAR_scores > threshold], LIDAR_scores[LIDAR_scores > threshold])
            
            score = scores[scores > threshold]
    times.append(time.time()-start)
    print(pred)
    preds.append(pred)  
    get_TruePositive(gt_bboxes_list[i], pred, score, i, "torch", iou)
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

talker([0])