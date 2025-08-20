from Fusion_code.mink_resnet_TR3D import TR3DMinkResNet
from Fusion_code.tr3d_neck import TR3DNeck
import torch.nn.functional as F

import os
import random
import re
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from pytorch3d.ops import knn_points
import sys
import time

sys.path.insert(1,'/home/jetson/theadams/')

from votenet.models.backbone_module import Pointnet2Backbone
from votenet.models.voting_module import VotingModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def to_batch_dim(ft, ct):
    batches = ct[:,0]
    batch_size = batches.max().item() + 1
    nr_features = torch.bincount(batches)
    max_nr_features = torch.max(nr_features)
    
    return_coords = torch.zeros([batch_size, max_nr_features , 3], device=ct.device)
    return_features = torch.zeros([batch_size, 128, max_nr_features], device=ft.device)
    
    for b in range(batch_size):
        mask = batches == b
        selected_feats = ft[mask]
        selected_coords = ct[mask][:, 1:]
        
        
        return_features[b, :, :selected_feats.shape[0]] = selected_feats.T
        return_coords[b, :selected_coords.shape[0], :] = selected_coords
    return return_features, return_coords, nr_features

def to_batch_output(ct, bb, cl, po):
    batches = ct[:,0]
    batch_size = batches.max().item() + 1
    
    out_bb, out_cl, out_po = [], [], []
    for b in range(batch_size):
        mask = batches == b
        bb_batch = bb[mask]
        cl_batch = cl[mask]
        po_batch = po[mask]
        
        out_bb.append(bb_batch)
        out_cl.append(cl_batch)
        out_po.append(po_batch)
    return [out_bb], [out_cl], [out_po]
    

def to_sparse_dim(ct, bt):
    B, C, N, _ = bt.shape
    device = bt.device
    batch_ids = ct[:, 0].long()  

    sorted_ids, sorted_idx = torch.sort(batch_ids)
    counts = torch.bincount(batch_ids, minlength=B)

    
    point_indices_sorted = torch.cat([torch.arange(c, device=device) for c in counts.tolist()])
    
    point_indices = torch.empty_like(point_indices_sorted)
    point_indices[sorted_idx] = point_indices_sorted

    
    out = bt[batch_ids, :, point_indices, 0]
    return out

def one_hot_encode(label_list):
    class_map = {
        3 : 0,
        4 : 1,
        5 : 2,
        6 : 3,
        7 : 4,
        8 : 5,
        9 : 6,
        10 : 7,
        11 : 8,
        12 : 9,
        14 : 10,
        16 : 11,
        24 : 12,
        28 : 13,
        33 : 14,
        34 : 15,
        36 : 16,
        39 : 17
    }
    return_tensor = torch.zeros([label_list.shape[0], 18], device = device)
    for i, l in enumerate(label_list):
        return_tensor[i, class_map[int(l)]] = 1
    
    return return_tensor


def normalize(tensor, new_min=0.0, new_max=1.0):
    t_min, t_max = tensor.min(), tensor.max()
    if t_min == t_max:
        return torch.full_like(tensor, new_min)
    return new_min + (tensor - t_min) * (new_max - new_min) / (t_max - t_min)

def normalize_points_color(points, color_mean=None):
    """
    Normalize RGB color channels in point cloud.
    
    Args:
        points (np.ndarray): (N, 6) array, with columns [x, y, z, r, g, b]
        color_mean (list or None): Mean values for r, g, b channels. 
                                   If None, compute from data.
    
    Returns:
        np.ndarray: Normalized points with shape (N, 6)
    """
    assert points.shape[1] >= 6, "Expected points with at least 6 dimensions (x, y, z, r, g, b)"
    
    colors = points[:, 3:6].astype(np.float32)
    
    if color_mean is None:
        color_mean = colors.mean(axis=0)
    
    
    normalized_colors = (colors - color_mean) / 255.0
    
    points[:, 3:6] = normalized_colors
    return points

def to_real_coords(voxels, voxel_size):
    return (voxels.float()) * voxel_size

def match_feat(vote_coords, vote_feat, tr3d_coords, tr3d_feat, lengths):
    _, ids, _ = knn_points(tr3d_coords[:,:,0:3], vote_coords[:,:,0:3],K=1, lengths1=lengths)
    index = ids.squeeze(-1).unsqueeze(1)  
    index = index.expand(-1, 256, -1)
    output = torch.gather(vote_feat, dim=2, index=index).to(device)
    
    return output
    


class RGBDBackbone(nn.Module):
    def __init__(self, weights_path="/home/jetson/theadams2/weights/TR3D.pth"):
        super().__init__()
        self.in_channels = 3
        self.depth = 34
        self.norm = 'batch'
        self.num_planes=(64, 128, 128, 128)
        self.backbone = TR3DMinkResNet(in_channels=self.in_channels, depth=self.depth, 
                                    norm=self.norm, num_planes=self.num_planes, pool=False)
        
        self.neck_in_channels=(64, 128, 128, 128) 
        self.neck_out_channels=128
        self.neck = TR3DNeck(in_channels=self.neck_in_channels, out_channels=self.neck_out_channels)
        
        # initialize weights of backbone and neck
        if os.path.exists(weights_path):
            
            state_dict = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
            backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict['state_dict'].items() if k.startswith('backbone.')}
            
            
            neck_state = {k.replace('neck.', ''): v for k, v in state_dict['state_dict'].items() if k.startswith('neck.')}
            self.neck.load_state_dict(neck_state, strict=True)
            self.backbone.load_state_dict(backbone_state, strict=True)
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")

    def forward(self,input_rgb):
        features = self.backbone(input_rgb)
        modified_features = self.neck(features)
        return modified_features


class LiDARBackbone(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=1, num_proposal=128, vote_factor=1, sampling='vote_fps', 
        backbone_path='/home/jetson/theadams2/weights/votenet_backbone.pth',
        neck_path='/home/jetson/theadams2/weights/votenet_neck.pth'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling
        

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting (Neck)
        self.vgen = VotingModule(self.vote_factor, 256)

        # initialize weights of backbone and neck
        if os.path.exists(neck_path) and os.path.exists(backbone_path):
            neck_state = torch.load(neck_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
            backbone_state = torch.load(backbone_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
            self.backbone_net.load_state_dict(backbone_state, strict=True)
            self.vgen.load_state_dict(neck_state, strict=True)
            print(f"Loaded pretrained weights from {backbone_path} and {neck_path}")
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {backbone_path} or {neck_path}")

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        
        
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features
        

        
        

        return end_points['vote_features'], end_points['vote_xyz']

class GatedFusion(nn.Module):
    """
    Implements the Gating Operation between RGB and LiDAR features as per the provided diagram.
    """
    def __init__(self, in_channels_rgb, in_channels_lidar):
        super(GatedFusion, self).__init__()
        # Individual MLPs for RGB and LiDAR features
        self.mlp_rgb = nn.Sequential(
            nn.Conv2d(in_channels_rgb, in_channels_rgb, kernel_size=1),
            nn.ReLU()
        )
        
        self.mlp_lidar = nn.Sequential(
            nn.Conv2d(in_channels_lidar, in_channels_lidar, kernel_size=1),
            nn.ReLU()
        )
        
        # MLP for computing gating weights
        self.mlp_weight = nn.Sequential(
            nn.Conv2d(in_channels_rgb + in_channels_lidar, in_channels_rgb + in_channels_lidar, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_features, lidar_features):
        processed_rgb = self.mlp_rgb(rgb_features)
        processed_lidar = self.mlp_lidar(lidar_features)
    
        combined = torch.cat((processed_rgb, processed_lidar), dim=1)  # No detach

        gating_weights = self.mlp_weight(combined)

        #hard_coded to avoid graphical confusion in tensor board
        
        
        rgb_shape = 128
        lidar_shape = 256
        
        split_tensors = torch.split(gating_weights, [rgb_shape, lidar_shape], dim=1)
        
        
        
        
        rgb_weight = split_tensors[0]
        lidar_weight = split_tensors[1]
    
        gated_rgb = rgb_features * rgb_weight
        gated_lidar = lidar_features * lidar_weight
    
        return torch.cat((gated_rgb, gated_lidar), dim=1)


class TR3DHead(nn.Module):
    def __init__(self, in_channels, num_reg_outs, num_classes, voxel_size = 0.02, 
                 pts_center_threshold=10, weights_path="/home/jetson/theadams2/weights/TR3D.pth"):
        super().__init__()
        self.voxel_size = voxel_size
        self.pts_center_threshold = pts_center_threshold

        self.conv_reg = ME.MinkowskiConvolution(
            in_channels, num_reg_outs, kernel_size=1, bias=True, dimension=3
        )
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels, num_classes, kernel_size=1, bias=True, dimension=3
        )

        # Init
        nn.init.normal_(self.conv_reg.kernel, std=0.01)
        nn.init.normal_(self.conv_cls.kernel, std=0.01)
        nn.init.constant_(self.conv_cls.bias, -torch.log(torch.tensor((1 - 0.01) / 0.01)))
        
    def forward_single(self, x):
        reg_out = self.conv_reg(x).features
        cls_out = self.conv_cls(x).features

        reg_distance = torch.exp(reg_out[:, 3:6])
        reg_angle = reg_out[:, 6:] if reg_out.shape[1] > 6 else None
        bbox_pred = torch.cat([reg_out[:, :3], reg_distance, reg_angle], dim=1) if reg_angle is not None else torch.cat([reg_out[:, :3], reg_distance], dim=1)

        return bbox_pred, cls_out, x.coordinates[:, 1:] * self.voxel_size

    def forward(self, sparse_tensor_list):
        all_bbox_preds, all_cls_preds, all_points = [], [], []
        for x in sparse_tensor_list:
            bbox_pred, cls_pred, points = self.forward_single(x)
            all_bbox_preds.append(bbox_pred)
            all_cls_preds.append(cls_pred)
            all_points.append(points)
        return all_bbox_preds, all_cls_preds, all_points


class CombinedModel(nn.Module):
    def __init__(self, rgb_backbone, lidar_backbone, voxel_size, n_classes=18, 
                 weights_path="/home/jetson/theadams2/weights/TR3D.pth"):
        super().__init__()
        self.voxel_size = voxel_size
        self.rgb_backbone = rgb_backbone
        self.lidar_backbone = lidar_backbone
        self.gated_fusion = GatedFusion(in_channels_rgb=128, in_channels_lidar=256).to(device)
        
        self.detection_head = TR3DHead(128 + 256, 6, 18).to(device)
        
        state_dict = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        
        head_state = {k.replace('head.', ''): v for k, v in state_dict['state_dict'].items() if k.startswith('head.')}
        head_state.pop("conv_reg.kernel", None)
        head_state.pop("conv_cls.kernel", None)
        
        self.detection_head.load_state_dict(head_state, strict=False)
        print(f"Loaded pretrained weights from {weights_path}")


    def forward(self, rgb_input, lidar_input):
        torch.cuda.synchronize()
        start = time.time()

        rgb_tensors = self.rgb_backbone(rgb_input)  

        lidar_features, lidar_coords = self.lidar_backbone(lidar_input)  

        tensors_head = []
        loop_start = time.time()
        for rgb_st in rgb_tensors:
            start = time.time()
            rgb_f = rgb_st.features
            rgb_c = rgb_st.coordinates
            
            map_key = rgb_st.coordinate_map_key
            manager = rgb_st.coordinate_manager
            
            rgb_features, rgb_coords, size_per_batch = to_batch_dim(rgb_f, rgb_c)
            
            
            rgb_real_coords = to_real_coords(rgb_coords, self.voxel_size)
            
            matched_lidar = match_feat(lidar_coords, lidar_features, rgb_real_coords, rgb_features, size_per_batch.to(device))
            rgb_features = rgb_features.unsqueeze(3)
            matched_lidar = matched_lidar.unsqueeze(3)
            
            fused_features = self.gated_fusion(rgb_features, matched_lidar)  # Apply Gating
            
            fused_features = to_sparse_dim(rgb_c, fused_features)
            
            
            sparse_fused = ME.SparseTensor(features=fused_features,
                                           coordinate_map_key=map_key,
                                           coordinate_manager=manager)
            
            tensors_head.append(sparse_fused)
        
        
        bb, cl, po = self.detection_head(tensors_head)
        bb, cl, po = to_batch_output(torch.cat([tensors_head[0].coordinates,tensors_head[1].coordinates]), 
                                     torch.cat(bb), torch.cat(cl), torch.cat(po))
        bb = torch.cat(bb[0])
        cl = torch.cat(cl[0])
        po = torch.cat(po[0])
        cls_thresholds = torch.max(torch.sigmoid(cl), dim=1, keepdim=True)[0][:,0]
        return bb, cl, po, cls_thresholds
    
class ConcatModel(nn.Module):
    def __init__(self, rgb_backbone, lidar_backbone, voxel_size, n_classes=18, 
                 weights_path="/home/jetson/theadams2/weights/TR3D.pth"):
        super().__init__()
        self.voxel_size = voxel_size
        self.rgb_backbone = rgb_backbone
        self.lidar_backbone = lidar_backbone
        
        self.detection_head = TR3DHead(128 + 256, 6, 18)
        
        state_dict = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        head_state = {k.replace('head.', ''): v for k, v in state_dict['state_dict'].items() if k.startswith('head.')}
        head_state.pop("conv_reg.kernel", None)
        head_state.pop("conv_cls.kernel", None)
        self.detection_head.load_state_dict(head_state, strict=False)
        print(f"Loaded pretrained weights from {weights_path}")


    def forward(self, rgb_input, lidar_input):
        rgb_tensors = self.rgb_backbone(rgb_input)  
        
        rgb_coords = [t.coordinates for t in rgb_tensors]
        rgb_coords = torch.cat(rgb_coords)
        
        rgb_feats = [t.features for t in rgb_tensors]
        rgb_feats = torch.cat(rgb_feats)
        
        lidar_features, lidar_coords = self.lidar_backbone(lidar_input)  
        
        rgb_features, _, _ = to_batch_dim(rgb_feats, rgb_coords)
        
        pad_amount = rgb_features.size(2) - lidar_features.size(2)
        
        padded_lidar = F.pad(lidar_features, (0, pad_amount))
        fused_feats = torch.cat([rgb_features, padded_lidar], dim=1)
        fused_feats = fused_feats.unsqueeze(3)
        
        feats = to_sparse_dim(rgb_coords, fused_feats)
        
        
        
        
        
        
             
        sparse_fused = ME.SparseTensor(features=feats, coordinates=rgb_coords, device=device)
        
        
        bb, cl, po = self.detection_head([sparse_fused])
        
        bb, cl, po = to_batch_output(sparse_fused.coordinates, 
                                     torch.cat(bb), torch.cat(cl), torch.cat(po))
        
        bb = torch.cat(bb[0])
        cl = torch.cat(cl[0])
        po = torch.cat(po[0])
        cls_thresholds = torch.max(torch.sigmoid(cl), dim=1, keepdim=True)[0][:,0]
        return bb, cl, po, cls_thresholds
    
if __name__ == "__main__":
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

    voxel_size = 0.2
    votenet = LiDARBackbone(num_class=18, num_heading_bin=1, num_size_cluster=18, mean_size_arr = mean_sizes).to(device)
    TR3D = RGBDBackbone().to(device)

    test_fusion = CombinedModel(TR3D, votenet, voxel_size).to(device)
    