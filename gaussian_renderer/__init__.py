#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
    render_size=None,
    retain_grad=False,
    cam_rot_delta=None,
    cam_trans_delta=None,
    get_flag=False,
    metric_map=None,
    train_cameras=None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros([pc.get_xyz.shape[0], 6], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height) if render_size is None else render_size[0],
        image_width=int(viewpoint_camera.image_width) if render_size is None else render_size[1],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        get_flag=get_flag,
        metric_map=metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image, accum_metric_counts = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            theta=cam_rot_delta,
            rho=cam_trans_delta)
    else:
        rendered_image, radii, depth_image, accum_metric_counts = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            theta=cam_rot_delta,
            rho=cam_trans_delta)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        if not viewpoint_camera.is_test_view:
            exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
            rendered_image = (
                torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(
                    2, 0, 1
                )
                + exposure[:3, 3, None, None]
            )
        else:
            # The exposure for the test script is interpolated in the spatial domain
            c2w_test = torch.inverse(viewpoint_camera.world_view_transform.T)
            for i in range(len(train_cameras)):
                # Use Rodrigues' formula to measure the distance between rotation matrices, consider Euclidean distance,
                # select the k nearest train cameras to c2w_test, and weight-average the exposure based on the inverse of the distance.
                # This avoids adding test images to the training set.
                c2w_train = torch.inverse(train_cameras[i].world_view_transform.T)
                rot_diff = c2w_test[:3, :3] @ c2w_train[:3, :3].T
                angle = torch.acos(
                    torch.clamp((torch.trace(rot_diff) - 1) / 2, -1.0, 1.0)
                )
                trans_diff = c2w_test[:3, 3] - c2w_train[:3, 3]
                dist = (
                    torch.norm(trans_diff) + angle * 0.1
                )  # The weights for rotation and translation are adjustable
                if i == 0:
                    dists = dist.unsqueeze(0)
                    exps = pc.get_exposure_from_name(
                        train_cameras[i].image_name
                    ).unsqueeze(0)
                else:
                    dists = torch.cat([dists, dist.unsqueeze(0)], dim=0)
                    exps = torch.cat(
                        [
                            exps,
                            pc.get_exposure_from_name(
                                train_cameras[i].image_name
                            ).unsqueeze(0),
                        ],
                        dim=0,
                    )
            k = min(5, len(train_cameras))
            dists_k, idxs = torch.topk(dists, k, largest=False)
            weights = 1.0 / (dists_k + 1e-8)
            weights = weights / weights.sum()
            exposure = (exps[idxs].T * weights).T.sum(dim=0)
            
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
        "accum_metric_counts": accum_metric_counts
    }
    
    return out