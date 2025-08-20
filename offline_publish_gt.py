# Different ROS nodes for publishing bounding boxes, points etc.
import rospy
import sys
import os
import json
from load_JSON import get_bboxes
import numpy as np
from sensor_msgs.msg import PointCloud2, CameraInfo
from std_msgs.msg import Header
import struct

sys.path.insert(1,'/home/jetson/theadams2')
from ros_numpy import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler
from lookup_table import color_list, gt_classes, pred_classes


LIDAR_dir = "/home/jetson/theadams2/unique_dataset/sequences/_robot_top_3d_laser_points"
RGBD_dir = "/home/jetson/theadams2/dataset/sequences/_robot_front_rgbd_camera_depth_color_points"
gt_file = "/home/jetson/theadams2/Cropped_LIDAR_Dataset-v0.1.json"

config_path = '/home/jetson/theadams2/mmdetection3d/configs/votenet/votenet_head-iouloss_8xb8_scannet-3d.py'
checkpoint_path = '/home/jetson/theadams2/weights/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth'

pub_GT = None
pub_PR = None
pub_GT_PC = None
pub_PC = None

def publish_bounding_boxes(bboxes, labels, frame_id="robot_base_footprint"):
    marker_array = MarkerArray()
    for i, bbox in enumerate(bboxes):
        if type(bbox) == dict:
            x = bbox['x']
            y = bbox['y']
            z = bbox['z']
            dx = bbox['dx']
            dy = bbox['dy']
            dz = bbox['dz']
            yaw = 0
            gt_class = gt_classes[bbox['class_id']]
            if gt_class == "chair": 
                r = 0
                g = 0
                b = 255
            elif gt_class == "bookshelf": 
                r, g, b = 230, 200, 0
            else:
                r = 100
                g = 255
                b = 100
            a = 0.3
            isDict = True
        else:
            r, g, b = color_list[labels[i]] 
            a = 0.3
            if len(bbox) == 7:
                x, y, z, dx, dy, dz, yaw = bbox
            else:
                x, y, z, dx, dy, dz = bbox
                yaw = 0
            isDict=False
        """
        # FULL BOXES
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bbox"
        marker.id = i * 2
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(9.9)
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z + 0.74
        marker.scale.x = dx
        marker.scale.y = dy
        marker.scale.z = dz

        q = quaternion_from_euler(0,0,yaw, "sxyz")
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
		
        marker.color.r = r / 255.0
        marker.color.g = g / 255.0
        marker.color.b = b / 255.0
        marker.color.a = a
        
        marker_array.markers.append(marker)"""

        # OUTLINE
        outline_marker = Marker()
        outline_marker.header.frame_id = frame_id
        outline_marker.header.stamp = rospy.Time.now()
        outline_marker.ns = "bbox_outline"
        outline_marker.id = i * 2 + 1   # Odd IDs for outlines
        outline_marker.type = Marker.LINE_LIST
        outline_marker.action = Marker.ADD
        outline_marker.lifetime = rospy.Duration(9.9)
        outline_marker.color.r = r / 255.0
        outline_marker.color.g = g / 255.0
        outline_marker.color.b = b / 255.0
        outline_marker.color.a = 1.0
        outline_marker.scale.x = 0.01   # Line thickness
        
        # Cube corner points (slightly larger so lines aren't hidden)
        half_x = dx/2 + 0.005
        half_y = dy/2 + 0.005
        half_z = dz/2 + 0.005
        cx, cy, cz = x, y, z + 0.74

        corners = [
            ( cx - half_x, cy - half_y, cz - half_z ),
            ( cx + half_x, cy - half_y, cz - half_z ),
            ( cx + half_x, cy + half_y, cz - half_z ),
            ( cx - half_x, cy + half_y, cz - half_z ),
            ( cx - half_x, cy - half_y, cz + half_z ),
            ( cx + half_x, cy - half_y, cz + half_z ),
            ( cx + half_x, cy + half_y, cz + half_z ),
            ( cx - half_x, cy + half_y, cz + half_z ),
        ]

        edges = [
            (0,1),(1,2),(2,3),(3,0), # bottom face
            (4,5),(5,6),(6,7),(7,4), # top face
            (0,4),(1,5),(2,6),(3,7)  # vertical edges
        ]

        for start, end in edges:
            outline_marker.points.append(Point(*corners[start]))
            outline_marker.points.append(Point(*corners[end]))
        
        marker_array.markers.append(outline_marker)
    return marker_array

def publish_gt_bounding_boxes(bboxes, labels, frame_id="robot_base_footprint"):
    marker_array = MarkerArray()
    class_names = {v: k for k,v in pred_classes.items()}
    for i, bbox in enumerate(bboxes):
        x = bbox['x']
        y = bbox['y']
        z = bbox['z']
        dx = bbox['dx']
        dy = bbox['dy']
        dz = bbox['dz']
        yaw = bbox['yaw']
        rgb = color_list[class_names[gt_classes[bbox['class_id']]]]
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bbox"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z + 0.74
        marker.scale.x = dx
        marker.scale.y = dy
        marker.scale.z = dz
        #print(dx," ", dy," ", dz)

        q = quaternion_from_euler(0,0,yaw, "sxyz")
        #print(q)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
		
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.4
        
        marker_array.markers.append(marker)
    return marker_array

def publish_points(file):
    input_data = np.fromfile(file,dtype=np.float32)
    input_data = input_data.reshape(-1,6)
    new_data = np.full((input_data.shape[0]), -1, dtype=[('x',np.float32),
                                                ('y', np.float32),
                                                ('z',np.float32),
                                                ('intensity',np.int32)])
    new_data[:]['x'] = input_data[:,0]
    new_data[:]['y'] = input_data[:,1]
    new_data[:]['z'] = input_data[:,2] + 0.74
    new_data[:]['intensity'] = input_data[:,3]

    return point_cloud2.array_to_pointcloud2(new_data.T, rospy.Time.now(), 'robot_base_footprint')


def rgb_to_float(rgb_array):
    float_arr = np.zeros(len(rgb_array), dtype=np.float32)
    for i, element in enumerate(rgb_array):
        r, g, b = element
        bits = (b << 16) | (g << 8) | r
        s = struct.pack('>l', bits)
        float_arr[i] = struct.unpack('>f', s)[0]
    
    return float_arr


def publish_colored_points(input_data):
    input_data = np.fromfile(input_data, dtype=np.float32).reshape(-1,6)
    new_data = np.full((input_data.shape[0]), -1, dtype=[('x',np.float32),
                                                ('y', np.float32),
                                                ('z',np.float32),
                                                ('rgb',np.float32)])
    rgb_array = np.full((input_data.shape[0]), -1, dtype=[('r',np.uint8),
                                                ('g', np.uint8),
                                                ('b',np.uint8)])
    rgb_array['r'] = input_data[:,5]
    rgb_array['g'] = input_data[:,4]
    rgb_array['b'] = input_data[:,3]
    rgb_data = point_cloud2.merge_rgb_fields(rgb_array)
    new_data[:]['x'] = input_data[:,0]
    new_data[:]['y'] = input_data[:,1]
    new_data[:]['z'] = input_data[:,2] + 0.74
    new_data[:]['rgb'] = rgb_data

    return point_cloud2.array_to_pointcloud2(new_data, rospy.Time.now(), 'robot_base_footprint')


def talker():
    rospy.init_node('votenet_test', anonymous=True)

    pub_GT = rospy.Publisher('/detection/Ground_Truth', MarkerArray, queue_size=1)
    pub_PC = rospy.Publisher('/detection/points', PointCloud2, queue_size=1)
    pub_RPC = rospy.Publisher('/detection/RGBD_points', PointCloud2, queue_size=1)

    files_LIDAR = sorted([f"{LIDAR_dir}/{f}" for f in os.listdir(LIDAR_dir) if f.endswith('.bin')])
    files_RGBD = sorted([f"{RGBD_dir}/{f}" for f in os.listdir(RGBD_dir) if f.endswith('.bin')])
    with open(gt_file) as f:
        data = json.load(f)

    frames = data['dataset']['samples'][0]['labels']['ground-truth']['attributes']['frames']
    gt_bboxes_list = get_bboxes(frames)
    rate = rospy.Rate(0.1) # 1hz
    i = 0
    #indexes = [15,34,41]
    indexes = [45]

    while not rospy.is_shutdown():
        markers = publish_gt_bounding_boxes(gt_bboxes_list[indexes[i]], None)
        points = publish_points(files_LIDAR[indexes[i]])
        RGBD_points = publish_colored_points(files_RGBD[indexes[i]])
        pub_PC.publish(points)
        pub_GT.publish(markers)
        pub_RPC.publish(RGBD_points)
        print(f"Publishing: {indexes[i]}")
        i += 1
        if i >= len(indexes):
            i=0
        rate.sleep()

def publish_corners(corner_array, is_pred, frame_id="robot_base_footprint"):
    marker_array = MarkerArray()
    for i,corner in enumerate(corner_array):
        x,y,z = corner
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bbox"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        #print(dx," ", dy," ", dz)

        q = quaternion_from_euler(0,0,0, "sxyz")
        #print(q)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        if is_pred:
            marker.color.r = color_list[i][0]
            marker.color.g = color_list[i][1]
            marker.color.b = color_list[i][2]
        else:
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 255
        marker.color.a = 1
        
        marker_array.markers.append(marker)
    return marker_array

def corner_talker(c_list):
    global pub_GT
    global pub_PR
    global pub_GT_PC
    global pub_PC
    if pub_GT == None and pub_PR == None: 
        rospy.init_node('corner_test', anonymous=True)

        pub_GT = rospy.Publisher('/detection/groundtruth', MarkerArray, queue_size=1)
        pub_PR = rospy.Publisher('/detection/preds', MarkerArray, queue_size=1)
        pub_GT_PC = rospy.Publisher('/detection/GT_IOU_corners', MarkerArray, queue_size=1)
        pub_PC = rospy.Publisher('/detection/P_IOU_corners', MarkerArray, queue_size=1)
    rate = rospy.Rate(0.2)
    i=0
    while not rospy.is_shutdown() and i < len(c_list):
        gt_box = c_list[i][0]
        pred_box = c_list[i][1]
        gt_corners = c_list[i][2]
        p_corners = c_list[i][3]
        iou = c_list[i][-1]
        print(iou)
        markers = publish_bounding_boxes([gt_box], None)
        print([pred_box[-1]])
        pred = publish_bounding_boxes([pred_box[0:7]], [int(pred_box[-1])])
        corners = publish_corners(p_corners, True)
        g_corners = publish_corners(gt_corners, False)
        pub_GT_PC.publish(g_corners)
        pub_PC.publish(corners)
        pub_GT.publish(markers)
        pub_PR.publish(pred)
        print(f"Publishing: {i, c_list[i][-1][0][0]}")
        i += 1
        rate.sleep()


def get_camera_info():
    cam_info = CameraInfo()
    cam_info.header.frame_id = "robot_front_rgbd_camera_color_optical_frame"
    cam_info.height = 480
    cam_info.width = 640
    cam_info.distortion_model = "plumb_bob"
    cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    cam_info.K = [606.704833984375, 0.0, 326.779296875,
                  0.0, 606.4002075195312, 244.65054321289062,
                  0.0, 0.0, 1.0]
    cam_info.R = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
    cam_info.P = [606.704833984375, 0.0, 326.779296875, 0.0,
                  0.0, 606.4002075195312, 244.65054321289062, 0.0,
                  0.0, 0.0, 1.0, 0.0]
    cam_info.binning_x = 0
    cam_info.binning_y = 0

    cam_info.roi.x_offset = 0
    cam_info.roi.y_offset = 0
    cam_info.roi.height = 0
    cam_info.roi.width = 0
    cam_info.roi.do_rectify = False
    return cam_info

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass