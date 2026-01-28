import os
import uuid
import json
from pathlib import Path
from typing import NamedTuple
import numpy as np
import torch
from torch import nn
from scene import GaussianModel
from simple_knn._C import distCUDA2
from utils.robotic_utils import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    sh_rotation,
)


class CamNamePoseInfo(NamedTuple):
    image_name: str
    R: np.array
    T: np.array
    pose: np.array  # Note: the output from anySplat is c2w


def cat_sparse_points_into_gaussians(pipe, gaussians, colmap_xyz, colmap_rgb):
    """
    Concatenate COLMAP sparse points into Gaussians:
      - Duplicate existing attributes (scaling/opacity/rotation/features_rest, etc.) n_new times
      - Change the xyz of the duplicated segment to colmap_xyz
      - Change the features_dc of the duplicated segment to colmap_rgb
    Shape auto-adaptation: features_dc can be either [N,3,1] or [N,1,3]
    """
    device = gaussians._xyz.device
    dtype = gaussians._xyz.dtype

    xyz_np = np.asarray(colmap_xyz, dtype=np.float32)
    rgb_np = np.asarray(colmap_rgb, dtype=np.float32)

    # If it is 0~255, convert to 0~1
    if rgb_np.max() > 1.5:
        rgb_np = rgb_np / 255.0

    new_xyz = torch.from_numpy(xyz_np).to(device=device, dtype=dtype)  # [K,3]
    new_rgb = torch.from_numpy(rgb_np).to(device=device, dtype=dtype)  # [K,3]
    n_new = new_xyz.shape[0]

    # --- Take a template and copy attributes ---
    def repeat_like(t, k):
        # Use the 0th as a template, repeat k times
        return t[:1].expand(k, *t.shape[1:]).clone()

    # tmpl_scaling   = repeat_like(gaussians._scaling,   n_new)   # [...,]

    dist2 = torch.clamp_min(
        distCUDA2(torch.from_numpy(np.asarray(xyz_np)).float().cuda()), 0.0000001
    )
    tmpl_scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    tmpl_opacity = repeat_like(gaussians._opacity, n_new)  # [...,]
    tmpl_rotation = repeat_like(gaussians._rotation, n_new)  # [K,4] quaternion
    tmpl_frest = repeat_like(
        gaussians._features_rest, n_new
    )  # [K, ?, 3] (your current implementation)
    tmpl_fdc = repeat_like(gaussians._features_dc, n_new)  # [K, 3,1] or [K,1,3]

    # --- Replace the xyz / fdc of the template segment with sfm's xyz / rgb ---
    # xyz direct replacement
    new_xyz_block = new_xyz

    if tmpl_fdc.dim() == 3 and tmpl_fdc.shape[1:] == (3, 1):
        new_fdc_block = new_rgb.unsqueeze(-1)  # [K,3,1]
    elif tmpl_fdc.dim() == 3 and tmpl_fdc.shape[1:] == (1, 3):
        new_fdc_block = new_rgb.unsqueeze(1)  # [K,1,3]
    else:
        if tmpl_fdc.size(-1) == 3:
            # e.g. [N,1,3] / [N,?,3]
            shape_front = list(tmpl_fdc.shape[1:-1])
            new_fdc_block = (
                new_rgb.view(n_new, *([1] * len(shape_front)), 3)
                .expand(n_new, *shape_front, 3)
                .clone()
            )
        else:
            raise ValueError(f"Unsupported features_dc shape: {tuple(tmpl_fdc.shape)}")

    # --- Concatenate back ---
    gaussians._xyz = nn.Parameter(torch.cat([gaussians._xyz, new_xyz_block], dim=0))
    gaussians._scaling = nn.Parameter(
        torch.cat([gaussians._scaling, tmpl_scaling], dim=0)
    )
    gaussians._opacity = nn.Parameter(
        torch.cat([gaussians._opacity, tmpl_opacity], dim=0)
    )
    gaussians._rotation = nn.Parameter(
        torch.cat([gaussians._rotation, tmpl_rotation], dim=0)
    )
    gaussians._features_rest = nn.Parameter(
        torch.cat([gaussians._features_rest, tmpl_frest], dim=0)
    )
    gaussians._features_dc = nn.Parameter(
        torch.cat([gaussians._features_dc, new_fdc_block], dim=0)
    )
    return gaussians


def anySplat(dataset, opt, pipe):
    import os
    import sys
    from pathlib import Path

    import torch

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

    from anySplat.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
    from anySplat.model.encoder.anysplat import EncoderAnySplatCfg

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
    from scene.colmap_loader import read_extrinsics_text, read_points3D_text, qvec2rotmat

    cameras_extrinsic_file = os.path.join(dataset.source_path, "sparse/0", "images.txt")
    SFM_pts_file = os.path.join(dataset.source_path, "sparse/0", "points3D.txt")

    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    xyz, rgb, _ = read_points3D_text(SFM_pts_file)  # points from colmap
    imgs = []
    cams_unsorted = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec).reshape(-1, 1)
        c2w = np.concatenate([R, T], axis=1)  # Note: the output from anySplat is c2w
        c2w = np.linalg.inv(
            np.concatenate([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        )
        cams_unsorted.append(CamNamePoseInfo(image_name=extr.name, R=R, T=T, pose=c2w))
    cams = sorted(cams_unsorted.copy(), key=lambda x: x.image_name)

    # image_names = os.listdir(images_dir)
    with open(f"{dataset.source_path}/train_test_split.json", "r") as f:
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
    interval = int(len(imgs) / 40) + 1  # process 40 training images
    imgs = imgs[::interval]
    slam_c2ws = slam_c2ws[::interval]

    slam_c2ws = np.concatenate(slam_c2ws, axis=0)
    images = [
        process_image(image_name) for id, image_name in enumerate(imgs) if id < 60
    ]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape

    # Run Inference
    gs, pred_context_pose = model.inference((images + 1) * 0.5)

    interval = pipe.FF_downsample
    gaussians = GaussianModel(3, opt.optimizer_type)
    gaussians._xyz = gs.means[0, ::interval, ...]
    gaussians._scaling = gaussians.scaling_inverse_activation(
        gs.scales[0, ::interval, ...] * 0.5
    )  # The scale of ff GS is too large, need to shrink it
    gaussians._opacity = gaussians.inverse_opacity_activation(
        gs.opacities[0, ::interval, ...][..., None]
    )
    gaussians._rotation = gs.rotations[0, ::interval, ...]
    gaussians._features_dc = (
        gs.harmonics[0, ::interval, :, :1].transpose(1, 2).contiguous()
    )
    gaussians._features_rest = (
        gs.harmonics[0, ::interval, :, 1:].transpose(1, 2).contiguous()
    )
    gaussians.active_sh_degree = int(gs.harmonics.shape[-1] ** 0.5 - 1)
    del gs, model
    torch.cuda.empty_cache()

    # gaussians.save_ply("/home/zzy/lib/siggraph_asia/tmp.ply")

    pred_all_extrinsic = pred_context_pose["extrinsic"][0]  # [N_nums, 4, 4]
    pred_all_intrinsic = pred_context_pose["intrinsic"][0]  # [N_nums, 3, 3]

    transform, stats, gaussians_aligned = align(
        gaussians=gaussians, anysplat_traj=pred_all_extrinsic, slam_traj=slam_c2ws
    )
    
    gaussians_aligned = cat_sparse_points_into_gaussians(
        pipe,
        gaussians_aligned,
        colmap_xyz=xyz,  # xyz returned by read_points3D_text
        colmap_rgb=rgb,  # rgb returned by read_points3D_text
    )

    return gaussians_aligned, pred_all_extrinsic, pred_all_intrinsic


def align(gaussians: GaussianModel, anysplat_traj, slam_traj, ply_path=None):
    """
    Align AnySplat trajectory to SLAM trajectory, apply the estimated similarity
    transform to the Gaussian model, and export the updated point cloud.
    """
    from tnt_eval.registration import (
        _trajectory_to_pose_tensor,
        estimate_similarity_transform,
    )

    if gaussians is None:
        raise ValueError("Gaussian model is required for alignment.")

    transform, stats = estimate_similarity_transform(anysplat_traj, slam_traj)

    rotation = torch.tensor(stats.get("rotation"), device=gaussians._xyz.device).float()
    translation = torch.tensor(
        stats.get("translation"), device=gaussians._xyz.device
    ).float()
    scale = torch.tensor(stats.get("scale"), device=gaussians._xyz.device).float()

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
    features_extra = sh_rotation(
        features_extra.reshape((features_extra.shape[0], 3, -1)),
        gaussians._features_dc,
        rotation,
    )
    gaussians._features_rest.data.copy_(
        features_extra.reshape((features_extra.shape[0], -1, 3))
    )
    del features_extra

    return transform, stats, gaussians