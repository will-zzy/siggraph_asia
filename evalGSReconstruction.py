

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from scene.gaussian_model import GaussianModel, GaussianModel_origin
import sys
from train_dash import eval
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from gaussian_renderer import prefilter_voxel, render, render_origin
from argparse import ArgumentParser, Namespace
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    # print(pc.use_feat_bank)
    # if pc.use_feat_bank:
    #     cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
    #     bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

    #     ## multi-resolution feat
    #     feat = feat.unsqueeze(dim=-1)
    #     feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
    #         feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
    #         feat[:,::1, :1]*bank_weight[:,:,2:]
    #     feat = feat.squeeze(dim=-1) # [n, c] 这里scaffold-gs只用了rgb，我们希望包含dc和sh


    if pc.add_opacity_dist or pc.add_cov_dist or pc.add_color_dist: # anchor如果不变的话这两个变量可以缓存
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    else:
        
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
        
        
    if pc.appearance_dim > 0:
        cam_idx = torch.full((feat.size(0),), viewpoint_camera.uid,
                             dtype=torch.long, device=anchor.device)
        appearance = pc.get_appearance(cam_idx)   

    # get offset's opacity
    if pc.add_opacity_dist:
        # neural_opacity = chunked(pc.get_opacity_mlp, cat_local_view) # anchor较少时用chunk反而会下降
        neural_opacity = pc.get_opacity_mlp(cat_local_view).float() # [N, k]
    else:
        # neural_opacity = chunked(pc.get_opacity_mlp, cat_local_view_wodist)
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist).float()

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    neural_opacity_flat = neural_opacity.reshape(-1, 1)
    mask = (neural_opacity > 0.0)
    # mask = mask.view(-1)
    mask = (neural_opacity_flat > 0.0).squeeze(1)
    idx  = mask.nonzero(as_tuple=False).squeeze(1)  

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            # color = chunked(pc.get_color_mlp, torch.cat([cat_local_view, appearance], dim=1))
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1)).float()
        else:
            # color = chunked(pc.get_color_mlp, torch.cat([cat_local_view_wodist, appearance], dim=1))
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1)).float()
    else:
        if pc.add_color_dist:
            # color = chunked(pc.get_color_mlp, cat_local_view)
            color = pc.get_color_mlp(cat_local_view).float()
        else:
            # color = chunked(pc.get_color_mlp, cat_local_view_wodist)
            color = pc.get_color_mlp(cat_local_view_wodist).float()
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        # scale_rot = chunked(pc.get_cov_mlp, cat_local_view)
        scale_rot = pc.get_cov_mlp(cat_local_view).float()
    else:
        # scale_rot = chunked(pc.get_cov_mlp, cat_local_view_wodist)
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist).float()
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    # offsets = grid_offsets.view([-1, 3]) # [mask]
    # 6) offsets / anchors / scaling 的“按需展开 + gather”
    N, k = grid_offsets.shape[:2]
    # (a) offsets => [N*k, 3]
    offsets_flat = grid_offsets.reshape(-1, 3)
    anchor_rep = anchor.repeat_interleave(k, dim=0)  
    scaling_rep = grid_scaling.repeat_interleave(k, dim=0) # [N*k, 6]
    opacity = neural_opacity_flat[idx]                     # [M, 1]
    color   = color.index_select(0, idx)                   # [M, 3]
    sr_sel  = scale_rot.index_select(0, idx)               # [M, 7]
    off_sel = offsets_flat.index_select(0, idx)            # [M, 3]
    anc_sel = anchor_rep.index_select(0, idx)              # [M, 3]
    scl_sel = scaling_rep.index_select(0, idx)    
    
    
    scaling = scl_sel[:, 3:] * torch.sigmoid(sr_sel[:, :3])    # [M, 3]
    rot     = pc.rotation_activation(sr_sel[:, 3:7])           # [M, 4] or whatever
    # offsets -> xyz
    off_scaled = off_sel * scl_sel[:, :3]
    xyz = anc_sel + off_scaled                                  # [M, 3]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity_flat, mask
    else:
        return xyz, color, opacity, scaling, rot
    
    

        
if __name__ == "__main__":
    parser = ArgumentParser(description="transforming")
    parser.add_argument('--load_ply', type=str, required=True)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * (i + 1) for i in range(30)])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--useFF", type=bool, default=False)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--log_file", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args = parser.parse_args(sys.argv[1:])
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    scaffold_gs = GaussianModel(n_offsets=dataset.n_offsets)
    vanilla_gs = GaussianModel_origin(4, "sparse_adam")
    scene = Scene(dataset, scaffold_gs, pipe, shuffle=False)
    scaffold_gs.load_mlp_checkpoints(args.load_ply)
    scaffold_gs.load_ply_sparse_gaussian(os.path.join(args.load_ply, "point_cloud.ply"))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    path = os.path.join(scene.model_path, "global_transform.txt")
    with open(path, "r") as f:
        lines = f.readlines()
    # 转成 float tensor
    rows = [list(map(float, line.strip().split())) for line in lines]
    global_transform = torch.tensor(rows, dtype=torch.float32)
    
    
    with torch.no_grad():
        eval(scene, render, render_origin, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), -1, 0, global_transform=global_transform)