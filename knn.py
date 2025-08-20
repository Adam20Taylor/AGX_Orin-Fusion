import torch
import os
import numpy as np
import time
from pytorch3d.ops import knn_points

RGBD_dir = "/home/jetson/theadams2/dataset/sequences/_robot_front_rgbd_camera_depth_color_points"
LIDAR_dir = "/home/jetson/theadams2/unique_dataset/sequences/_robot_top_3d_laser_points"

def do_knn(files_RGBD, files_LIDAR):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(files_RGBD) is list:
        RGBD_np_list = [np.fromfile(file, dtype=np.float32).reshape(-1, 6) for file in files_RGBD]
        LIDAR_np_list = [np.fromfile(file, dtype=np.float32).reshape(-1, 6) for file in files_LIDAR]
    else:
        RGBD_np_list = [np.fromfile(files_RGBD, dtype=np.float32).reshape(-1, 6)]
        LIDAR_np_list = [np.fromfile(files_LIDAR, dtype=np.float32).reshape(-1, 6)]

    RGBD_np = np.zeros((len(RGBD_np_list), max(len(item) for item in RGBD_np_list),6))
    LIDAR_np = np.zeros((len(LIDAR_np_list), max(len(item) for item in LIDAR_np_list),6))

    for i in range(len(RGBD_np_list)):
        nr_RGBD_points = len(RGBD_np_list[i])
        nr_LIDAR_points = len(LIDAR_np_list[i])
        RGBD_np[i,:nr_RGBD_points,0] = RGBD_np_list[i][:,0]
        RGBD_np[i,:nr_RGBD_points,1] = RGBD_np_list[i][:,1]
        RGBD_np[i,:nr_RGBD_points,2] = RGBD_np_list[i][:,2]
        RGBD_np[i,:nr_RGBD_points,3] = RGBD_np_list[i][:,3]
        RGBD_np[i,:nr_RGBD_points,4] = RGBD_np_list[i][:,4]
        RGBD_np[i,:nr_RGBD_points,5] = RGBD_np_list[i][:,5]
        
        LIDAR_np[i,:nr_LIDAR_points,0] = LIDAR_np_list[i][:,0]
        LIDAR_np[i,:nr_LIDAR_points,1] = LIDAR_np_list[i][:,1]
        LIDAR_np[i,:nr_LIDAR_points,2] = LIDAR_np_list[i][:,2]

    tensor_RGBD = torch.tensor(RGBD_np, dtype=torch.float32).to(device)
    tensor_LIDAR = torch.tensor(LIDAR_np, dtype=torch.float32).to(device)
    _, ids, _ = knn_points(tensor_LIDAR[:,:,0:3], tensor_RGBD[:,:,0:3],K=5)
    #print(tensor_LIDAR.shape)
    return avg_RGB(tensor_RGBD,tensor_LIDAR, ids)


def avg_RGB(points_RGBD, points_LIDAR, K_nearest_ids):
    for frame in range(points_RGBD.shape[0]):
        rgbd = points_RGBD[frame]          # (N_rgbd, 6)
        lidar = points_LIDAR[frame]        # (N_lidar, 6)
        neighbors = K_nearest_ids[frame]   # (N_lidar, K)

        neighbor_RGBs = rgbd[neighbors][:, :, 3:6]   # (N_lidar, K, 3)

        mean_RGBs = neighbor_RGBs.mean(dim=1)        # (N_lidar, 3)

        lidar[:, 3:6] = mean_RGBs
        points_LIDAR[frame] = lidar
    return [frame.cpu().numpy() for frame in points_LIDAR]
            

if __name__ == "__main__":
    files_LIDAR = sorted([f"{LIDAR_dir}/{f}" for f in os.listdir(LIDAR_dir) if f.endswith('.bin')])
    files_RGBD = sorted([f"{RGBD_dir}/{f}" for f in os.listdir(RGBD_dir) if f.endswith('.bin')])    

    print(do_knn([files_RGBD[0]], [files_LIDAR[0]]))
