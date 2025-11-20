import os
import numpy as np
import torch
import torchvision
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
import json
import uuid
from torch import nn
from PIL import Image
from simple_knn._C import distCUDA2
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.camera_utils import update_pose, update_pose_by_global
# from gaussian_renderer import render, network_gui
from gaussian_renderer import prefilter_voxel, render, network_gui, render_origin, render_simp
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import GaussianModel_origin
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.schedule_utils import TrainingScheduler
from utils.robotic_utils import sh_rotation, matrix_to_quaternion, quaternion_to_matrix, load_ply, save_ply
from utils.fast_utils import sampling_cameras, compute_gaussian_score_fastgs

import cv2
import csv
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

UPLOAD_IMG = False
from typing import NamedTuple
class CamNamePoseInfo(NamedTuple):
    image_name: str
    R: np.array
    T: np.array
    pose: np.array # 注意anySplat出来的c2w
    
    
def _o3d_from_np(points, estimate_normals=False, voxel_size=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if estimate_normals:
        # 邻域半径与邻居数可按体素尺度调
        if voxel_size is None:
            voxel_size = max(1e-3, float(np.linalg.norm(points.max(0)-points.min(0)))/200.0)
        radius = 2.5 * voxel_size
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=32))
        pcd.orient_normals_consistent_tangent_plane(10)
    return pcd

def _downsample_auto(pcd, voxel_size):
    import open3d as o3d
    return pcd.voxel_down_sample(voxel_size=max(1e-6, float(voxel_size)))

def _compute_scale_rot_trans_from_4x4(T):
    """
    Open3D 的 with_scaling 估计会把 sR 放在左上 3x3。
    这里把 s、R、t 拆出来：R 正交化，s 取列范数平均。
    """
    M = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    # 估计尺度：三列范数的平均（对正交 R，列范数应全为 s）
    s = float(np.mean(np.linalg.norm(M, axis=0)))
    # 防御性处理
    if s <= 0:
        s = 1.0
    R = M / s
    # 正交化（极分解），避免数值漂移
    U, _, Vt = np.linalg.svd(R)
    R = (U @ Vt)
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return s, R, t

def _global_ransac_with_fpfh(src_down, tgt_down, voxel_size):
    import open3d as o3d
    # FPFH 特征
    radius_normal = 2.5 * voxel_size
    radius_feature = 5.0 * voxel_size
    src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=32))
    tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=32))
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=64)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=64)
    )

    distance_threshold = 1.5 * voxel_size
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(2.0 * voxel_size),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 1000)
    )
    return result.transformation

def align_icp_with_sfm(
    gaussians,
    colmap_xyz,                 # numpy [N,3], 来自 points3D.txt
    voxel_div=200,              # 体素尺度 = 对角线 / voxel_div
    use_point_to_plane=True,    # 是否用点到平面 ICP
    with_scaling=True,          # 是否估计尺度（Sim(3)）
    ply_path=None
):
    """
    用 COLMAP SFM 稀疏点与 AnySplat 高斯中心做 Sim(3) 对齐（RANSAC 预对齐 + ICP 精化）。
    返回 (4x4 变换矩阵, stats, gaussians_aligned)
    """
    # --- 取源（AnySplat 高斯中心）与目标（COLMAP 稀疏点） ---
    src_xyz = gaussians._xyz.detach().cpu().numpy().astype(np.float64)  # [M,3]
    tgt_xyz = np.asarray(colmap_xyz, dtype=np.float64)                   # [N,3]
    if src_xyz.size == 0 or tgt_xyz.size == 0:
        raise ValueError("Empty source or target points for ICP alignment.")

    # 自动体素尺度
    diag = float(np.linalg.norm(np.max(tgt_xyz, axis=0) - np.min(tgt_xyz, axis=0)))
    voxel_size = max(diag / float(voxel_div), 1e-4)

    import open3d as o3d
    src_pcd = _o3d_from_np(src_xyz, estimate_normals=False)
    tgt_pcd = _o3d_from_np(tgt_xyz, estimate_normals=True, voxel_size=voxel_size)

    # 下采样（提升稳健性）
    src_down = _downsample_auto(src_pcd, voxel_size)
    tgt_down = _downsample_auto(tgt_pcd, voxel_size)

    # ---- 全局 RANSAC 初始位姿（不带尺度） ----
    init_T = _global_ransac_with_fpfh(src_down, tgt_down, voxel_size)

    # ---- ICP 精化（可带尺度） ----
    if use_point_to_plane and len(np.asarray(tgt_down.normals)) == len(np.asarray(tgt_down.points)):
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        # 如果需要 with_scaling 的 ICP，Open3D 没有 point-to-plane+with_scaling，退回 point-to-point
        if with_scaling:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=with_scaling)

    icp_threshold = 3.0 * voxel_size
    result_icp = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down,
        icp_threshold,
        init_T,
        est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    T = result_icp.transformation  # 4x4

    # ---- 拆解 s, R, t ----
    s, R, t = _compute_scale_rot_trans_from_4x4(T)
    stats = {"rotation": R, "translation": t.reshape(3), "scale": s, "icp_fitness": result_icp.fitness, "icp_rmse": result_icp.inlier_rmse}

    # ---- 应用到高斯 ----
    rotation = torch.tensor(R, device=gaussians._xyz.device, dtype=torch.float32)
    translation = torch.tensor(t.reshape(1,3), device=gaussians._xyz.device, dtype=torch.float32)
    scale = torch.tensor([s], device=gaussians._xyz.device, dtype=torch.float32)

    # 位置
    xyz = gaussians._xyz
    xyz = scale * (xyz @ rotation.t()) + translation
    gaussians._xyz.data.copy_(xyz); del xyz

    # 尺度
    scaling = gaussians._scaling
    scaling = scaling + torch.log(scale * 2.0)
    gaussians._scaling.data.copy_(scaling); del scaling

    # 旋转
    rots = gaussians._rotation
    rot_matrices = quaternion_to_matrix(rots)
    new_rot_matrices = rotation @ rot_matrices
    rots = matrix_to_quaternion(new_rot_matrices)
    rots = rots / torch.norm(rots, dim=-1, keepdim=True)
    gaussians._rotation.data.copy_(rots); del rots, rot_matrices, new_rot_matrices

    # SH 旋转
    features_extra = gaussians._features_rest
    features_extra = sh_rotation(features_extra.reshape((features_extra.shape[0],3,-1)), gaussians._features_dc, rotation)
    gaussians._features_rest.data.copy_(features_extra.reshape((features_extra.shape[0],-1,3))); del features_extra

    # 可选：保存可视化或 .ply
    if ply_path is not None:
        gaussians.save_ply(ply_path)

    # 可选：保存轨迹对齐图（此处用点云对齐，不再画轨迹）
    return T, stats, gaussians
def cat_sparse_points_into_gaussians(pipe, gaussians, colmap_xyz, colmap_rgb):
    """
    将 COLMAP 稀疏点拼接进高斯：
      - 复制现有各属性（scaling/opacity/rotation/features_rest 等）n_new 份
      - 将复制出的那一段的 xyz 改为 colmap_xyz
      - 将复制出的那一段的 features_dc 改为 colmap_rgb
    形状自适配：features_dc 既可能是 [N,3,1] 也可能是 [N,1,3]
    """
    device = gaussians._xyz.device
    dtype  = gaussians._xyz.dtype

    xyz_np = np.asarray(colmap_xyz, dtype=np.float32)
    rgb_np = np.asarray(colmap_rgb, dtype=np.float32)

    # 若是 0~255，转换到 0~1
    if rgb_np.max() > 1.5:
        rgb_np = rgb_np / 255.0

    new_xyz = torch.from_numpy(xyz_np).to(device=device, dtype=dtype)               # [K,3]
    new_rgb = torch.from_numpy(rgb_np).to(device=device, dtype=dtype)               # [K,3]
    n_new   = new_xyz.shape[0]

    # --- 取一个模板，复制属性 ---
    def repeat_like(t, k):
        # 以第 0 个为模板，repeat k 次
        return t[:1].expand(k, *t.shape[1:]).clone()

    # tmpl_scaling   = repeat_like(gaussians._scaling,   n_new)   # [...,]
    
    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz_np)).float().cuda()), 0.0000001)
    tmpl_scaling = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    tmpl_opacity   = repeat_like(gaussians._opacity,   n_new)   # [...,]
    tmpl_rotation  = repeat_like(gaussians._rotation,  n_new)   # [K,4] 四元数
    tmpl_frest     = repeat_like(gaussians._features_rest, n_new)  # [K, ?, 3]（你当前实现）
    tmpl_fdc       = repeat_like(gaussians._features_dc,   n_new)  # [K, 3,1] 或 [K,1,3]

    # --- 将 template 段的 xyz / fdc 替换为 sfm 的 xyz / rgb ---
    # xyz 直接替换
    new_xyz_block = new_xyz

    if tmpl_fdc.dim() == 3 and tmpl_fdc.shape[1:] == (3,1):
        new_fdc_block = new_rgb.unsqueeze(-1)          # [K,3,1]
    elif tmpl_fdc.dim() == 3 and tmpl_fdc.shape[1:] == (1,3):
        new_fdc_block = new_rgb.unsqueeze(1)           # [K,1,3]
    else:
        if tmpl_fdc.size(-1) == 3:
            # e.g. [N,1,3] / [N,?,3]
            shape_front = list(tmpl_fdc.shape[1:-1])
            new_fdc_block = new_rgb.view(n_new, *([1]*len(shape_front)), 3).expand(n_new, *shape_front, 3).clone()
        else:
            raise ValueError(f"Unsupported features_dc shape: {tuple(tmpl_fdc.shape)}")

    # --- 拼接回去 ---
    if pipe.useScaffold:
        gaussians._xyz          = torch.cat([gaussians._xyz,          new_xyz_block], dim=0)
        gaussians._scaling      = torch.cat([gaussians._scaling,      tmpl_scaling],  dim=0)
        gaussians._opacity      = torch.cat([gaussians._opacity,      tmpl_opacity],  dim=0)
        gaussians._rotation     = torch.cat([gaussians._rotation,     tmpl_rotation], dim=0)
        gaussians._features_rest= torch.cat([gaussians._features_rest,tmpl_frest],    dim=0)
        gaussians._features_dc  = torch.cat([gaussians._features_dc,  new_fdc_block], dim=0)
    else:
        gaussians._xyz          = nn.Parameter(torch.cat([gaussians._xyz,          new_xyz_block], dim=0))
        gaussians._scaling      = nn.Parameter(torch.cat([gaussians._scaling,      tmpl_scaling],  dim=0))
        gaussians._opacity      = nn.Parameter(torch.cat([gaussians._opacity,      tmpl_opacity],  dim=0))
        gaussians._rotation     = nn.Parameter(torch.cat([gaussians._rotation,     tmpl_rotation], dim=0))
        gaussians._features_rest= nn.Parameter(torch.cat([gaussians._features_rest,tmpl_frest],    dim=0))
        gaussians._features_dc  = nn.Parameter(torch.cat([gaussians._features_dc,  new_fdc_block], dim=0))
    return gaussians
def anySplat(dataset, opt, pipe):
    
    from pathlib import Path
    import torch
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for loading local AnySplat checkpoints. "
            "Please install it with `pip install safetensors`."
        ) from exc
    from anySplat.model.model.anysplat import AnySplat
    from utils.image_utils import process_image

    
    # use AnySplat pre-trained model
    ckpt_root = Path("./anySplat/ckpt")
    config_path = ckpt_root / "config.json"
    weights_path = ckpt_root / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"AnySplat config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"AnySplat weights not found: {weights_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    encoder_cfg_dict = config_dict.get("encoder_cfg")
    decoder_cfg_dict = config_dict.get("decoder_cfg")
    if encoder_cfg_dict is None or decoder_cfg_dict is None:
        raise ValueError(f"Invalid AnySplat config structure: {config_path}")

    from anySplat.model.encoder.anysplat import EncoderAnySplatCfg
    from anySplat.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg

    vggt_override = os.environ.get("ANY_SPLAT_VGGT_WEIGHTS")
    if vggt_override:
        encoder_cfg_dict["pretrained_weights"] = vggt_override

    encoder_cfg = EncoderAnySplatCfg(**encoder_cfg_dict)
    decoder_cfg = DecoderSplattingCUDACfg(**decoder_cfg_dict)
    model = AnySplat(encoder_cfg, decoder_cfg)

    state_dict = load_file(str(weights_path))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[AnySplat] Missing keys when loading weights: {missing_keys}")
    if unexpected_keys:
        print(f"[AnySplat] Unexpected keys when loading weights: {unexpected_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    
    
    
    
    # load SfM data, align, and cat SfM points
    images_dir = os.path.join(dataset.source_path, dataset.images)
    from scene.colmap_loader import read_extrinsics_text, qvec2rotmat, read_points3D_text
    cameras_extrinsic_file = os.path.join(dataset.source_path, "sparse/0", "images.txt")
    SFM_pts_file = os.path.join(dataset.source_path, "sparse/0", "points3D.txt")
    
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    xyz, rgb, _ = read_points3D_text(SFM_pts_file) # 来自colmap的点
    imgs = []
    cams_unsorted = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec).reshape(-1, 1)
        c2w = np.concatenate([R, T], axis=1) # 注意anySplat出来的是c2w
        c2w = np.linalg.inv(np.concatenate([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0))
        cams_unsorted.append(CamNamePoseInfo(image_name=extr.name, R=R, T=T, pose=c2w))
    cams = sorted(cams_unsorted.copy(), key = lambda x : x.image_name)
    
    # image_names = os.listdir(images_dir)
    with open(f"{dataset.source_path}/train_test_split.json", 'r') as f:
        train_test_exp = json.load(f)
    slam_c2ws = []
    for cam in cams:
        img_name = cam.image_name
        if img_name not in train_test_exp["train"]:
            continue
        image_path = os.path.join(images_dir, img_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(os.path.dirname(images_dir), "images", extr.name)
        imgs.append(os.path.join(image_path))
        slam_c2ws.append(cam.pose[None])
    interval = int(len(imgs) / 20) + 1 # 处理20张train
    imgs = imgs[::interval]
    slam_c2ws = slam_c2ws[::interval]
    
    slam_c2ws = np.concatenate(slam_c2ws, axis=0)
    images = [process_image(image_name) for id, image_name in enumerate(imgs) if id < 60]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape

    # Run Inference
    gs, pred_context_pose = model.inference((images + 1)*0.5)
    # gaussians.mean
    # downsample = 8
    # downsample_points = pipe.FF_downsample # for scaffold's anchor
    # interval = int(gs.means.shape[1] / downsample_points) + 1
    
    interval = pipe.FF_downsample
    gaussians = GaussianModel_origin(3, opt.optimizer_type)
    gaussians._xyz = gs.means[0, ::interval, ...]
    gaussians._scaling = gaussians.scaling_inverse_activation(gs.scales[0, ::interval, ...] * 0.3) # ff GS的scale太大了，要缩小一些
    gaussians._opacity = gaussians.inverse_opacity_activation(gs.opacities[0, ::interval, ...][..., None])
    gaussians._rotation = gs.rotations[0, ::interval, ...]
    gaussians._features_dc = gs.harmonics[0, ::interval, :, :1].transpose(1, 2).contiguous()
    gaussians._features_rest = gs.harmonics[0, ::interval, :, 1:].transpose(1, 2).contiguous()
    gaussians.active_sh_degree = int(gs.harmonics.shape[-1]**0.5 - 1)
    del gs, model
    torch.cuda.empty_cache()
    
    # gaussians.save_ply("/home/zzy/lib/siggraph_asia/tmp.ply")
    
    pred_all_extrinsic = pred_context_pose['extrinsic'][0] # [N_nums, 4, 4]
    pred_all_intrinsic = pred_context_pose['intrinsic'][0] # [N_nums, 3, 3]
    
    transform, stats, gaussians_aligned = align(gaussians=gaussians, anysplat_traj=pred_all_extrinsic, slam_traj=slam_c2ws)
    # transform, stats, gaussians_aligned = align_icp_with_sfm(
    #     gaussians=gaussians,
    #     colmap_xyz=xyz,          # points3D.txt 读到的 sfm 稀疏点
    #     voxel_div=200,           # 体素大小 = 场景对角线 / voxel_div，可按数据调
    #     use_point_to_plane=True, # 有法线时更稳；没有就设 False
    #     with_scaling=True        # 允许估计尺度（Sim(3)）
    # )
    gaussians_aligned = cat_sparse_points_into_gaussians(
    pipe,
    gaussians_aligned,
    colmap_xyz=xyz,   # read_points3D_text 返回的 xyz
    colmap_rgb=rgb    # read_points3D_text 返回的 rgb
    )
        
    return gaussians_aligned, pred_all_extrinsic, pred_all_intrinsic


def align(gaussians: GaussianModel_origin, anysplat_traj, slam_traj, ply_path=None):
    """
    Align AnySplat trajectory to SLAM trajectory, apply the estimated similarity
    transform to the Gaussian model, and export the updated point cloud.
    """
    from tnt_eval.registration import (
        estimate_similarity_transform,
        apply_similarity_transform_to_gaussians,
        _trajectory_to_pose_tensor,
    )

    if gaussians is None:
        raise ValueError("Gaussian model is required for alignment.")

    transform, stats = estimate_similarity_transform(anysplat_traj, slam_traj)
    
    
    # align_out = apply_similarity_transform_to_gaussians(
    #     gaussians, # {"scale": scale, "rotation_quaternion": rotation_matrix_to_quaternion(rotation)}
    #     transform,
    #     rotation=stats.get("rotation"),
    #     translation=stats.get("translation"),
    #     scale=stats.get("scale"),
    # )
    rotation=torch.tensor(stats.get("rotation"),device=gaussians._xyz.device).float()
    translation=torch.tensor(stats.get("translation"),device=gaussians._xyz.device).float()
    scale=torch.tensor(stats.get("scale"),device=gaussians._xyz.device).float()
    
    
    # rotation = transform_matrix[:3, :3]
    # translation = transform_matrix[None, :3, 3]
    # scale_check = (rotation @ rotation.T)[0, 0] ** 0.5
    # rotation_pure = rotation / scale_check
    xyz = gaussians._xyz
    xyz = scale * (xyz @ rotation.T) + translation
    gaussians._xyz.data.copy_(xyz)
    del xyz
    
    scaling = gaussians._scaling
    scaling = scaling + torch.log(scale * 2)
    gaussians._scaling.data.copy_(scaling)
    del scaling
    
    rots = gaussians._rotation
    rot_matrices = quaternion_to_matrix(rots)
    new_rot_matrices = rotation @ rot_matrices
    rots = matrix_to_quaternion(new_rot_matrices)
    rots = rots / torch.norm(rots, dim=-1, keepdim=True)
    gaussians._rotation.data.copy_(rots)
    del rots, rot_matrices, new_rot_matrices
    
    features_extra = gaussians._features_rest
    features_extra = sh_rotation(features_extra.reshape((features_extra.shape[0],3,-1)), gaussians._features_dc, rotation)
    gaussians._features_rest.data.copy_(features_extra.reshape((features_extra.shape[0],-1,3)))
    del features_extra

    # gaussian_new = GaussianModel_origin(4, "sparse_adam")
    # gaussian_new._xyz = xyz
    # gaussian_new._scaling = scaling
    # gaussian_new._rotation = rots
    # gaussian_new._opacity = gaussians._opacity
    # gaussian_new._features_dc = features_dc
    # gaussian_new._features_rest = features_extra
    
    
    # xyz_np = xyz.cpu().numpy()
    # features_dc_np = features_dc.reshape(-1, 3).cpu().numpy()
    # features_extra_np = features_extra.reshape(-1, 45).cpu().numpy()
    # opacities_np = opacities.cpu().numpy()
    # scales_np = scales.cpu().numpy()
    # rots_np = rots.cpu().numpy()

    if ply_path is not None:
        gaussians.save_ply(ply_path)

    # Prepare debug outputs (plot + point clouds) to inspect alignment quality.
    try:
        source_tensor = _trajectory_to_pose_tensor(anysplat_traj)
        target_tensor = _trajectory_to_pose_tensor(slam_traj)
    except Exception as exc:
        print(f"[align] Unable to convert trajectories for visualization: {exc}")
        return transform, stats

    rotation = stats.get("rotation")
    translation = stats.get("translation")
    scale = stats.get("scale")

    if rotation is None or translation is None or scale is None:
        print("[align] Missing rotation/translation/scale in stats; skip visualization.")
        return transform, stats

    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    scale = float(scale)

    source_centers = source_tensor[:, :3, 3].cpu().numpy()
    aligned_centers = (rotation @ source_centers.T).T
    aligned_centers = scale * aligned_centers + translation
    gt_centers = target_tensor[:, :3, 3].cpu().numpy()

    base_dir = Path(ply_path).parent if ply_path is not None else Path.cwd() / "align_outputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(ply_path).stem if ply_path is not None else "align"
    unique_tag = uuid.uuid4().hex[:8]
    prefix = f"{base_name}_{unique_tag}"

    png_path = base_dir / f"{prefix}_traj.png"
    # aligned_ply_path = base_dir / f"{prefix}_aligned_traj.ply"
    # gt_ply_path = base_dir / f"{prefix}_gt_traj.ply"

    # Save 3D trajectory plot.
    # try:
    #     import matplotlib
    #     import sys as _sys

    #     if "matplotlib.pyplot" not in _sys.modules:
    #         matplotlib.use("Agg")  # Ensure headless plotting.
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    #     fig = plt.figure(figsize=(6, 6))
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(
    #         aligned_centers[:, 0],
    #         aligned_centers[:, 1],
    #         aligned_centers[:, 2],
    #         label="aligned traj",
    #         color="tab:blue",
    #     )
    #     ax.plot(
    #         gt_centers[:, 0],
    #         gt_centers[:, 1],
    #         gt_centers[:, 2],
    #         label="gt traj",
    #         color="tab:orange",
    #     )
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("Trajectory Alignment")
    #     ax.legend()
    #     fig.tight_layout()
    #     fig.savefig(png_path, dpi=300)
    #     plt.close(fig)
    #     stats["trajectory_plot_png"] = str(png_path)
    # except Exception as exc:
    #     print(f"[align] Failed to save trajectory plot: {exc}")

    # # Save aligned and GT trajectories as point clouds.
    # try:
    #     import open3d as o3d

    #     aligned_pcd = o3d.geometry.PointCloud()
    #     aligned_pcd.points = o3d.utility.Vector3dVector(
    #         np.asarray(aligned_centers, dtype=np.float64)
    #     )
    #     o3d.io.write_point_cloud(str(aligned_ply_path), aligned_pcd)

    #     gt_pcd = o3d.geometry.PointCloud()
    #     gt_pcd.points = o3d.utility.Vector3dVector(
    #         np.asarray(gt_centers, dtype=np.float64)
    #     )
    #     o3d.io.write_point_cloud(str(gt_ply_path), gt_pcd)

    #     stats["aligned_traj_ply"] = str(aligned_ply_path)
    #     stats["gt_traj_ply"] = str(gt_ply_path)
    # except Exception as exc:
    #     print(f"[align] Failed to save trajectory point clouds: {exc}")

    return transform, stats, gaussians


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, log_file=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians = None
    if args.useFF:
        
        FF_gaussians, pred_all_extrinsic, pred_all_intrinsic = anySplat(dataset, opt, pipe)
        FF_gaussians.xyz_gradient_accum = torch.zeros((FF_gaussians.get_xyz.shape[0], 1), device=FF_gaussians.get_xyz.device)
        FF_gaussians.denom = torch.zeros((FF_gaussians.get_xyz.shape[0], 1), device=FF_gaussians.get_xyz.device)
        if pipe.useScaffold:
            # align()
            gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                  dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
            scene = Scene(dataset, gaussians, pipe, FF_gaussians, shuffle=False)
        else:
            gaussians = GaussianModel_origin(dataset.feat_dim, opt.optimizer_type)
            scene = Scene(dataset, gaussians, pipe, FF_gaussians, shuffle=False)
            gaussians = scene.gaussians
    else:
        if pipe.useScaffold:
            pass
            gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                  dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    #         anchor_xyz, pred_all_extrinsic, pred_all_intrinsic = anySplat(dataset, opt, pipe)
            scene = Scene(dataset, gaussians, pipe, shuffle=False)
        else:
    #         gaussians = GaussianModel_origin(dataset.feat_dim, opt.optimizer_type)
    #         scene = Scene(dataset, gaussians, pipe, shuffle=False)
            gaussians = GaussianModel_origin(dataset.feat_dim, opt.optimizer_type)
            scene = Scene(dataset, gaussians, pipe, shuffle=False)
    
    
    
    gaussians.training_setup(opt)
    if checkpoint:
        
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    # optim_start = torch.cuda.Event(enable_timing = True)
    # optim_end = torch.cuda.Event(enable_timing = True)
    # total_time = 0.0
    all_time=60.0
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Init DashGaussian scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians, 
                                  [cam.original_image for cam in scene.getTrainCameras()])
    render_scale = scheduler.get_res_scale(1)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 添加时间计数器，在iteration开始后开始计时
    eval_time = None
    time_save_iterations = [30, 60]  # 30秒和60秒时保存
    
    data_device = torch.device("cuda")
    
    
    
    
    cam_rot_delta = nn.Parameter(
        torch.zeros(3, requires_grad=True, device=data_device)
    )
    cam_trans_delta = nn.Parameter(
        torch.zeros(3, requires_grad=True, device=data_device)
    )
    global_transform = torch.eye(4, device=data_device)
    
    l = [ 
        {'params': [cam_rot_delta], 'lr': 0.00002, "name": "pose_rot_delta"}, # 0.00008
        {'params': [cam_trans_delta], 'lr': 0.00001, "name": "pose_trans_delta"}, # 0.00005
    ]
    
    pose_optimizer = torch.optim.Adam(l)
    
    start_time = time.time()
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        # vind = viewpoint_indices.pop(rand_idx)
        # pred_w2c = torch.inverse(pred_all_extrinsic[rand_idx, ...])
        # viewpoint_cam.R = pred_w2c[:3, :3].T
        # viewpoint_cam.T = pred_w2c[:3, 3]
        # viewpoint_cam.update_W2C = False
        # viewpoint_cam.update_W2I = False
        # viewpoint_cam.update_center = False

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        retain_grad = (iteration < opt.update_until and iteration >= 0)
        # Rescale GT image for DashGaussian
        gt_image = viewpoint_cam.original_image.cuda()
        if render_scale > 1:
            gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", 
                                                       recompute_scale_factor=True, antialias=True)[0]
        if pipe.useScaffold:
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_image.shape[-2:], visible_mask=voxel_visible_mask, retain_grad=retain_grad, cam_rot_delta=cam_rot_delta, cam_trans_delta=cam_trans_delta)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        else:
            render_pkg = render_origin(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_image.shape[-2:],retain_grad=retain_grad, cam_rot_delta=cam_rot_delta, cam_trans_delta=cam_trans_delta)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # if viewpoint_cam.alpha_mask is not None:
        #     alpha_mask = viewpoint_cam.alpha_mask.cuda()
        #     if render_scale > 1:
        #         alpha_mask = torch.nn.functional.interpolate(alpha_mask[None], scale_factor=1/render_scale, 
        #                                                 mode="bilinear", recompute_scale_factor=True, antialias=True)[0]
        #     image *= alpha_mask

        # Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0 # TODO: maybe add depth supervision
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable: # 加速收敛
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure *0.3
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if iteration % 200 == 0:
                image_write = image.permute(1,2,0).detach().cpu().numpy()
                image_write = (image_write * 255).astype("uint8")
                os.makedirs(f"{scene.model_path}/test/", exist_ok = True)
                cv2.imwrite(os.path.join(f"{scene.model_path}/test/", "iter{:06d}_{}.png".format(iteration, viewpoint_cam.image_name)), cv2.cvtColor(image_write, cv2.COLOR_RGB2BGR))

        
        
        
        
        
        
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                if pipe.useFF:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "N_GS": f"{gaussians._scaling.shape[0]}", "N_MAX": f"{scheduler.max_n_gaussian}", "R": f"{render_scale}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "N_GS": f"{gaussians._scaling.shape[0] * dataset.n_offsets}", "N_MAX": f"{scheduler.max_n_gaussian}", "R": f"{render_scale}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            ########## TODO 是否要删除test视角渲染？
            # iter_time = iter_start.elapsed_time(iter_end)
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
            ##########
                
            #  检查时间计数器，30秒和60秒时保存结果
            elapsed_time = time.time() - start_time
            case_name = dataset.source_path.split('/')[-1]
            # if elapsed_time >= 30 and 30 in time_save_iterations:
            #     eval_start_time = time.time()
            #     print(f"\n[ITER {iteration}] Evaluating Gaussians at 30 seconds")
            #     eval(case_name, scene, render, render_origin, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), iteration, 30, log_file, global_transform)
            #     eval_time = time.time() - eval_start_time
            #     time_save_iterations.remove(30)  # 移除已保存的时间点，避免重复保存
            # elif (eval_time != None and elapsed_time - eval_time >= 60 and 60 in time_save_iterations) or iteration==opt.iterations:
            #     print(f"\n[ITER {iteration}] Saving Gaussians at 60 seconds")
            #     all_time = elapsed_time - eval_time
            #     eval(case_name, scene, render, render_origin, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), iteration, 60, log_file, global_transform, all_time)
            #     scene.save(iteration)
            #     time_save_iterations.remove(60)  # 移除已保存的时间点，避免重复保存
            #     # 到60秒后退出训练
                
            #     sys.exit(0)
            if (elapsed_time >= 60 and 60 in time_save_iterations) or iteration==opt.iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians at 60 seconds")
                all_time = elapsed_time
                eval(pipe, case_name, scene, render, render_origin, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), iteration, 60, log_file, global_transform, all_time)
                scene.save(iteration)
                time_save_iterations.remove(60)  # 移除已保存的时间点，避免重复保存
                # 到60秒后退出训练
                
                sys.exit(0)



            if pipe.useScaffold: # 使用scaffold-GS
                # Densification
                if iteration < opt.densify_until_iter and iteration > opt.start_stat and (gaussians._scaling.shape[0] * dataset.n_offsets) < pipe.max_n_gaussian:
                    # Keep track of max radii in image-space for pruning
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.densification_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                    # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    #     # Apply DashGaussian primitive scheduler to control densification.
                    #     densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_anchor.shape[0], render_scale)
                    #     momentum_add = gaussians.prune_and_densify(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 
                    #                                                size_threshold, radii, densify_rate=densify_rate)
                    #     # Update max_n_gaussian
                    #     scheduler.update_momentum(momentum_add)
                    #     # Update render scale based on the DashGaussian resolution scheduler. 
                    #     render_scale = scheduler.get_res_scale(iteration)

                    # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    #     gaussians.reset_opacity()
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()

                # Optimizer step
                if iteration < opt.iterations:
                    # gaussians.exposure_optimizer.step()
                    # gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    # if use_sparse_adam:
                    #     visible = radii > 0
                    #     gaussians.optimizer.step(visible, radii.shape[0])
                    #     gaussians.optimizer.zero_grad(set_to_none = True)
                    # else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none = True)

                # optim_end.record()
                # torch.cuda.synchronize()
                # optim_time = optim_start.elapsed_time(optim_end)
                # total_time += (iter_time + optim_time) / 1e3

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            
            
            
            
            elif not pipe.useScaffold: # 使用原版3DGS表达
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # Apply DashGaussian primitive scheduler to control densification.
                        densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_xyz.shape[0], render_scale)
                        
                        sample_view_stack = scene.getTrainCameras().copy()
                        camlist = sampling_cameras(sample_view_stack)
                        
                        importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, dataset, opt, pipe, bg, DENSIFY=True, cam_rot_delta=cam_rot_delta, cam_trans_delta=cam_trans_delta)                    
                        gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)
                        
                        
                        
                        # momentum_add = gaussians.prune_and_densify(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 
                        #                                            size_threshold, radii, densify_rate=densify_rate)
                        # Update max_n_gaussian
                        # scheduler.update_momentum(momentum_add)
                        # Update render scale based on the DashGaussian resolution scheduler. 
                        # render_scale = scheduler.get_res_scale(iteration)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)                    
                    gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)

                # Optimizer step
                if iteration < opt.iterations:
                    # gaussians.exposure_optimizer.step()
                    # gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

                    # viewpoint_cam.pose_optimizer.step()
                    # viewpoint_cam.pose_optimizer.zero_grad(set_to_none = True)
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)
            
            if iteration > 500 and iteration % 300 == 0 and iteration < opt.update_until :
                # update_pose(viewpoint_cam)
                update_global=True
                for view in scene.getTrainCameras():
                    _, global_transform = update_pose(view, cam_trans_delta, cam_rot_delta, global_transform, update_global)
                    update_global=False
                    # update_pose(view)
                with open(os.path.join(scene.model_path, "global_transform.txt"), "w+") as f:
                    for row in global_transform.cpu().numpy():
                        f.write(" ".join([f"{x:.8f}" for x in row]) + "\n")
                cam_rot_delta.data.fill_(0)
                cam_trans_delta.data.fill_(0)
    with open(os.path.join(scene.model_path, "TRAIN_INFO"), "w+") as f:
        # f.write("Training Time: {:.2f} seconds, {:.2f} minutes\n".format(total_time, total_time / 60.))
        f.write("GS Number: {}\n".format(gaussians._scaling.shape[0]))
        
    
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
    
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})
        (pipe, background, scale_factor, SPARSE_ADAM_AVAILABLE, overide_color, train_test_exp) = renderArgs
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, pipe, background, 1.0, None)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if UPLOAD_IMG and tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    # image_write = image.permute(1,2,0).detach().cpu().numpy()
                    # image_write = (image_write * 255).astype("uint8")
                    # os.makedirs(f"{scene.model_path}/test/", exist_ok = True)
                    # cv2.imwrite(os.path.join(f"{scene.model_path}/test/", "iter{:06d}_{}_{}.png".format(iteration, config['name'], viewpoint.image_name)), cv2.cvtColor(image_write, cv2.COLOR_RGB2BGR))
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_scaling.shape[0], iteration)
        torch.cuda.empty_cache()
        
def eval(pipe, case_name, scene : Scene, renderFunc, renderFunc2, renderArgs, iteration: int, time: int, log_file=None, global_transform=None, all_time=None):
    torch.cuda.empty_cache()
    config = {'name': 'test', 'cameras' : scene.getTestCameras()}
    (pipe, background, scale_factor, SPARSE_ADAM_AVAILABLE, overide_color, train_test_exp) = renderArgs
    if config['cameras'] and len(config['cameras']) > 0:
        l1_test = 0.0
        psnr_test = 0.0
        lpips_test = 0.0
        ssim_test = 0.0
        for idx, viewpoint in enumerate(config['cameras']):
            
            transform_viewpoint = update_pose_by_global(viewpoint, global_transform)
            if pipe.useScaffold:
                voxel_visible_mask = prefilter_voxel(transform_viewpoint, scene.gaussians, pipe, background, 1.0, None)
                image = torch.clamp(renderFunc(transform_viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
            else:
                image = torch.clamp(renderFunc2(transform_viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0) 
            # try:
            #     voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, pipe, background, 1.0, None)
            #     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
            # except:
            #     image = torch.clamp(renderFunc2(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
            # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(transform_viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if train_test_exp:
                image = image[..., image.shape[-1] // 2:]
                gt_image = gt_image[..., gt_image.shape[-1] // 2:]
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            lpips_test += lpips_fn(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            image_write = image.permute(1,2,0).detach().cpu().numpy()
            image_write = (image_write * 255).astype("uint8")
            os.makedirs(f"{scene.model_path}/test/", exist_ok = True)
            cv2.imwrite(os.path.join(f"{scene.model_path}/test/", "{}_{}_iter{:06d}.png".format(config['name'], transform_viewpoint.image_name, iteration)), cv2.cvtColor(image_write, cv2.COLOR_RGB2BGR))
        
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        lpips_test /= len(config['cameras'])
        ssim_test /= len(config['cameras'])
        print("\n[ITER {}] Evaluating {}sec: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, time, l1_test, psnr_test, lpips_test, ssim_test))
        
        # 记录到文件
        # if log_file is not None:
        #     with open(log_file, 'a', newline='') as csvfile:
        #         log_writer = csv.writer(csvfile)
        #         if time == 30:
        #             log_writer.writerow([case_name, 30.0, f"{ssim_test:.4f}", f"{lpips_test:.4f}", f"{psnr_test:.4f}"])
        #         else:
        #             log_writer.writerow([case_name, f"{float(all_time):.1f}", f"{ssim_test:.4f}", f"{lpips_test:.4f}", f"{psnr_test:.4f}"])
        if log_file is not None:
            data = {}
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            t_val = f"{float(all_time):.1f}"

            psnr_val = f"{float(psnr_test):.4f}"

            data[case_name] = {
                "PSNR": psnr_val,
                "time": t_val
            }

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)   
            torch.cuda.empty_cache()
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
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
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.log_file)

    # All done
    print("\nTraining complete.")
