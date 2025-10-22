# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

import math
from typing import Sequence, Tuple, Union

from trajectory_io import read_trajectory, convert_trajectory_to_pointcloud
import copy
import numpy as np
import open3d as o3d
import torch
try:
    from open3d import registration
except:
    from open3d.pipelines import registration

MAX_POINT_NUMBER = 4e6


def _trajectory_to_pose_tensor(
    trajectory: Union[Sequence, np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    Convert a trajectory into a torch tensor of shape (N, 4, 4).
    Supports sequences of CameraPose, numpy arrays, or torch tensors.
    """
    if trajectory is None:
        raise ValueError("Trajectory cannot be None.")

    if isinstance(trajectory, torch.Tensor):
        if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected trajectory tensor of shape (N,4,4), got {tuple(trajectory.shape)}."
            )
        return trajectory.clone().float()

    poses_np = None
    if isinstance(trajectory, np.ndarray):
        poses_np = trajectory
    elif isinstance(trajectory, Sequence) and len(trajectory) > 0:
        sample = trajectory[0]
        if hasattr(sample, "pose"):
            poses_np = np.stack([np.asarray(t.pose) for t in trajectory], axis=0)
        else:
            poses_np = np.asarray(trajectory)
    else:
        raise ValueError("Unsupported trajectory type.")

    poses_np = np.asarray(poses_np)
    if poses_np.ndim != 3 or poses_np.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected trajectory array of shape (N,4,4), got {poses_np.shape}."
        )
    return torch.from_numpy(poses_np.astype(np.float32))


def estimate_similarity_transform(
    source_traj: Union[Sequence, np.ndarray, torch.Tensor],
    target_traj: Union[Sequence, np.ndarray, torch.Tensor],
) -> Tuple[np.ndarray, dict]:
    """
    Estimate a similarity transform (rotation, uniform scale, translation) that maps
    source trajectory poses onto target trajectory poses. The returned transform
    applies rotation first, followed by uniform scaling, and finally translation:
        x' = scale * R @ x + t
    """
    from utils.pose_eval import AlignPose

    source = _trajectory_to_pose_tensor(source_traj)
    target = _trajectory_to_pose_tensor(target_traj)

    if source.shape[0] < 3 or target.shape[0] < 3:
        raise ValueError("Need at least three poses in each trajectory.")

    if source.shape[0] != target.shape[0]:
        min_len = min(source.shape[0], target.shape[0])
        source = source[:min_len]
        target = target[:min_len]
    else:
        min_len = source.shape[0]

    aligned, rot, t, scale = AlignPose(est_pose=source, gt_pose=target)

    rot = np.asarray(rot, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    scale = float(scale)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot * scale
    transform[:3, 3] = t

    aligned_centers = aligned[:, :3, 3].cpu().numpy()
    gt_centers = target[:, :3, 3].cpu().numpy()
    errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)

    diagnostics = {
        "rotation": rot,
        "translation": t,
        "scale": scale,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_error": float(np.median(errors)),
        "count": int(min_len),
    }

    return transform, diagnostics


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion in (w, x, y, z) format.
    """
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        indices = np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2]])
        i = int(np.argmax(indices))
        if i == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def apply_similarity_transform_to_gaussians(
    gaussians, transform: np.ndarray, rotation: np.ndarray = None, translation: np.ndarray = None, scale: float = None
) -> dict:
    """
    Apply a similarity transform to a GaussianModel_origin instance in-place.

    Args:
        gaussians: The GaussianModel_origin containing xyz positions, scaling, and rotation.
        transform: 4x4 homogeneous matrix with optional uniform scale embedded.

    Returns:
        Dictionary with applied scale and rotation quaternion for reference.
    """
    import torch
    import torch.nn.functional as F

    if transform.shape != (4, 4):
        raise ValueError("Transform must be 4x4.")

    if rotation is None or translation is None or scale is None:
        linear = transform[:3, :3]
        translation = transform[:3, 3]
        det = np.linalg.det(linear)
        if det <= 0.0:
            raise ValueError("Similarity transform must have positive determinant.")
        scale = det ** (1.0 / 3.0)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Invalid scale extracted from transform.")
        rotation = linear / scale

    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    scale = float(scale)

    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0

    device = gaussians._xyz.device
    dtype = gaussians._xyz.dtype
    rotation_t = torch.tensor(rotation, dtype=dtype, device=device)
    translation_t = torch.tensor(translation, dtype=dtype, device=device)
    scale_t = torch.tensor(scale, dtype=dtype, device=device)

    with torch.no_grad():
        xyz = gaussians._xyz
        transformed_xyz = torch.matmul(xyz, rotation_t.T) * scale_t + translation_t
        gaussians._xyz.data.copy_(transformed_xyz)

        gaussians._scaling.data.add_(math.log(scale))

        local_rot = F.normalize(gaussians._rotation, dim=1)
        global_quat = torch.tensor(
            rotation_matrix_to_quaternion(rotation),
            dtype=dtype,
            device=device,
        )
        w0, x0, y0, z0 = global_quat
        w1, x1, y1, z1 = (
            local_rot[:, 0],
            local_rot[:, 1],
            local_rot[:, 2],
            local_rot[:, 3],
        )
        new_rot = torch.stack(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dim=1,
        )
        gaussians._rotation.data.copy_(F.normalize(new_rot, dim=1))

    return {"scale": scale, "rotation_quaternion": rotation_matrix_to_quaternion(rotation)}


def read_mapping(filename):
    mapping = []
    with open(filename, "r") as f:
        n_sampled_frames = int(f.readline())
        n_total_frames = int(f.readline())
        mapping = np.zeros(shape=(n_sampled_frames, 2))
        metastr = f.readline()
        for iter in range(n_sampled_frames):
            metadata = list(map(int, metastr.split()))
            mapping[iter, :] = metadata
            metastr = f.readline()
    return [n_sampled_frames, n_total_frames, mapping]


def gen_sparse_trajectory(mapping, f_trajectory):
    sparse_traj = []
    for m in mapping:
        sparse_traj.append(f_trajectory[int(m[1] - 1)])
    return sparse_traj


def trajectory_alignment(map_file, traj_to_register, gt_traj_col, gt_trans,
                         scene):
    traj_pcd_col = convert_trajectory_to_pointcloud(gt_traj_col)
    if gt_trans is not None:
        traj_pcd_col.transform(gt_trans)
    corres = o3d.utility.Vector2iVector(
        np.asarray(list(map(lambda x: [x, x], range(len(gt_traj_col))))))
    rr = registration.RANSACConvergenceCriteria()
    rr.max_iteration = 100000
    # rr.max_validation = 100000

    # in this case a log file was used which contains
    # every movie frame (see tutorial for details)
    if len(traj_to_register) > 1600 and map_file is not None:
        n_sampled_frames, n_total_frames, mapping = read_mapping(map_file)
        traj_col2 = gen_sparse_trajectory(mapping, traj_to_register)
        traj_to_register_pcd = convert_trajectory_to_pointcloud(traj_col2)
    else:
        print("Estimated trajectory will leave as it is, no sparsity op is performed!")
        traj_to_register_pcd = convert_trajectory_to_pointcloud(
            traj_to_register)
    randomvar = 0.0
    if randomvar < 1e-5:
        traj_to_register_pcd_rand = traj_to_register_pcd
    else:
        nr_of_cam_pos = len(traj_to_register_pcd.points)
        rand_number_added = np.asanyarray(traj_to_register_pcd.points) * (
            np.random.rand(nr_of_cam_pos, 3) * randomvar - randomvar / 2.0 + 1)
        list_rand = list(rand_number_added)
        traj_to_register_pcd_rand = o3d.geometry.PointCloud()
        for elem in list_rand:
            traj_to_register_pcd_rand.points.append(elem)

    # Rough registration based on aligned colmap SfM data
    reg = registration.registration_ransac_based_on_correspondence(
        traj_to_register_pcd_rand,
        traj_pcd_col,
        corres,
        0.2,
        registration.TransformationEstimationPointToPoint(True),
        6,
        criteria=rr,
    )
    return reg.transformation


def crop_and_downsample(
        pcd,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=0.01,
        trans=np.identity(4),
):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(trans)
    pcd_crop = crop_volume.crop_point_cloud(pcd_copy)
    if down_sample_method == "voxel":
        # return voxel_down_sample(pcd_crop, voxel_size)
        return pcd_crop.voxel_down_sample(voxel_size)
    elif down_sample_method == "uniform":
        n_points = len(pcd_crop.points)
        if n_points > MAX_POINT_NUMBER:
            ds_rate = int(round(n_points / float(MAX_POINT_NUMBER)))
            return pcd_crop.uniform_down_sample(ds_rate)
    return pcd_crop


def registration_unif(
    source,
    gt_target,
    init_trans,
    crop_volume,
    threshold,
    max_itr,
    max_size=4 * MAX_POINT_NUMBER,
    verbose=True,
):
    if verbose:
        print("[Registration] threshold: %f" % threshold)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = crop_and_downsample(source,
                            crop_volume,
                            down_sample_method="uniform",
                            trans=init_trans)
    t = crop_and_downsample(gt_target,
                            crop_volume,
                            down_sample_method="uniform")
    reg = registration.registration_icp(
        s,
        t,
        threshold,
        np.identity(4),
        registration.TransformationEstimationPointToPoint(True),
        registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg


def registration_vol_ds(
    source,
    gt_target,
    init_trans,
    crop_volume,
    voxel_size,
    threshold,
    max_itr,
    verbose=True,
):
    if verbose:
        print("[Registration] voxel_size: %f, threshold: %f" %
              (voxel_size, threshold))
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = crop_and_downsample(
        source,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
        trans=init_trans,
    )
    t = crop_and_downsample(
        gt_target,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
    )
    reg = registration.registration_icp(
        s,
        t,
        threshold,
        np.identity(4),
        registration.TransformationEstimationPointToPoint(True),
        registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg
