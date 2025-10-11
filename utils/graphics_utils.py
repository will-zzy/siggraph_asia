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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
#     translate = translate.to(R.device)
#     Rt = torch.zeros((4, 4), device=R.device)
#     # Rt[:3, :3] = R.transpose()
#     Rt[:3, :3] = R.T # 注意camera里的R是W2C.R的转置
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0

#     C2W = torch.linalg.inv(Rt)
#     cam_center = C2W[:3, 3]
#     cam_center = (cam_center + translate) * scale
#     # cam_center = (cam_center)
#     C2W[:3, 3] = cam_center
#     Rt = torch.linalg.inv(C2W)
#     return Rt
@torch.no_grad()
def getWorld2View2(R, t, translate=None, scale=1.0):
    """
    R: 你类里存的 R（按你的注释，是 W2C.R 的转置，即 C2W.R）
    t: 你类里存的 T（对应 W2C 的平移）
    返回：W2C 4x4（避免任何 linalg.inv）
    """
    device = R.device
    if translate is None:
        translate = torch.zeros(3, device=device)
    else:
        translate = translate.to(device)

    # C2W：R_c2w = R, c = -R @ t
    c = - R @ t  # 相机中心（世界系）

    # 应用平移/缩放
    c_new = (c + translate) * scale

    # 回到 W2C：R_w2c = Rᵀ, t' = -Rᵀ @ c_new
    R_w2c = R.T
    t_new = - R_w2c @ c_new

    W2C = torch.eye(4, device=device, dtype=R.dtype)
    W2C[:3, :3] = R_w2c
    W2C[:3, 3]  = t_new
    return W2C
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))