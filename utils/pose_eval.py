'''
Author: Diantao Tu
Date: 2024-03-07 10:40:33
'''

import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import median
from typing import Dict, List, Tuple, Union
import torch
import roma
import os


def RotationError(R1_list, R2_list):
    """Compute rotation error between two list of rotation matrix.
    """
    assert len(R1_list) == len(R2_list)
    errors = []
    for R1, R2 in zip(R1_list, R2_list):
        angle = (np.trace(np.dot(R1.T, R2)) - 1) / 2.0
        angle = np.clip(angle, -1, 1)
        errors.append(np.arccos(angle) * 180 / np.pi)
    return errors
    
'''
对齐两组旋转矩阵
gt_rotations: 真值旋转矩阵 [N, 3, 3]
est_rotations: 估计旋转矩阵 [N, 3, 3]
'''
def AlignRotations(gt_rotations: np.ndarray, est_rotations: np.ndarray):
    assert len(gt_rotations) == len(est_rotations)

    for _ in range(4):
        rand_idx = np.random.randint(0, len(gt_rotations))
        base_gt_rotation = gt_rotations[rand_idx].T
        base_est_rotation = est_rotations[rand_idx].T

        # gt_rotations = [np.dot(r, base_gt_rotation) for r in gt_rotations]
        # est_rotations = [np.dot(r, base_est_rotation) for r in est_rotations]
        gt_rotations = np.matmul(gt_rotations, base_gt_rotation)
        est_rotations = np.matmul(est_rotations, base_est_rotation)

        total_iter = 0
        while total_iter < 100:
            R_diff = np.einsum('...ji,...jk->...ik', est_rotations, gt_rotations)
            angle_axis = R.from_matrix(R_diff).as_rotvec()

            adjustment_median = np.median(angle_axis, axis=0)

            # Check if the adjustment is below the threshold
            if np.linalg.norm(adjustment_median) < 1e-5:
                break

            adjust_R = R.from_rotvec(adjustment_median).as_matrix()
            est_rotations = np.matmul(est_rotations, adjust_R)
            total_iter += 1

    return gt_rotations, est_rotations



def Umeyama_Alignment(x: np.ndarray, y: np.ndarray,
                      with_scale: bool = False) :
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    注意这个结果是把 x 变换到 y 的坐标系下, 即 y = scale * R @ x + t
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise ValueError("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise ValueError("Degenerate covariance rank, "
                                "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

'''
使用Umeyama算法对齐两个位姿
est_pose: 估计的位姿 [N, 4, 4]
gt_pose: 真值位姿 [N, 4, 4]
return: 对齐后的位姿 [N, 4, 4]
'''
def AlignPose(est_pose:torch.Tensor, gt_pose:torch.Tensor) -> torch.Tensor:
    est_centers = est_pose[:, :3, 3].cpu().numpy()  # N x 3
    gt_centers = gt_pose[:, :3, 3].cpu().numpy()  # N x 3

    rot, t, scale = Umeyama_Alignment(est_centers.T, gt_centers.T, True)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    # 对est_pose中的旋转部分进行变换        
    est_rotations = est_pose.cpu().numpy()
    est_rotations = np.matmul(T[None,...], est_rotations)   # [N, 4, 4]
    # 对est_pose中的平移部分进行变换
    est_centers = scale * (rot @ est_centers.T) + t.reshape(3,1)  # 3 x N
    est_centers = est_centers.T                    # N x 3
    # 把变换后的结果组合并保存
    aligned_pose = est_rotations
    aligned_pose[:, :3, 3] = est_centers
    return torch.from_numpy(aligned_pose).float().to(est_pose.device), rot, t, scale

def AlignPoseRansac(est_pose:torch.Tensor, gt_pose:torch.Tensor, ransac_iter:int=100) -> torch.Tensor:
    est_centers = est_pose[:, :3, 3].cpu().numpy()  # N x 3
    gt_centers = gt_pose[:, :3, 3].cpu().numpy()  # N x 3

    best_rot, best_t, best_scale = np.eye(3), np.zeros(3), 1.0
    max_inlier = 0

    for _ in range(ransac_iter):
        rand_idx = np.random.choice(len(gt_centers), 3, replace=False)
        rot, t, scale = Umeyama_Alignment(est_centers[rand_idx].T, gt_centers[rand_idx].T, True)
        aligned_est_centers = scale * (rot @ est_centers.T) + t.reshape(3,1)
        aligned_est_centers = aligned_est_centers.T
        error = np.linalg.norm(aligned_est_centers - gt_centers, axis=1)
        num_inliers = np.sum(error < 10.0)
        if max_inlier < num_inliers:
            best_rot, best_t, best_scale = rot, t, scale
            max_inlier = num_inliers

    T = np.eye(4)
    T[:3, :3] = best_rot
    T[:3, 3] = best_t
    # 对est_pose中的旋转部分进行变换
    est_rotations = est_pose.cpu().numpy()
    est_rotations = np.matmul(T[None,...], est_rotations)   # [N, 4, 4]
    # 对est_pose中的平移部分进行变换
    est_centers = best_scale * (best_rot @ est_centers.T) + best_t.reshape(3,1)  # 3 x N
    est_centers = est_centers.T                    # N x 3
    # 把变换后的结果组合并保存
    aligned_pose = est_rotations
    aligned_pose[:, :3, 3] = est_centers
    return torch.from_numpy(aligned_pose).float().to(est_pose.device)


@torch.no_grad()
def evaluate_pose_kitti(est_pose:torch.Tensor, gt_pose:torch.Tensor, align=False) -> Dict[str, float]:
    """
    用KITTI官方的方式评估位姿, 比较不同轨迹长度下的相对位姿误差. 适用于户外大场景.
    """
    # KITTI evaluation 
    subsequence_length = [100, 200, 300, 400, 500, 600, 700, 800]
    # compute the trajectory length for each frame using the ground truth
    gt_centers = gt_pose[:, :3, 3].cpu().numpy()    # N x 3
    dist = np.linalg.norm(gt_centers[1:] - gt_centers[:-1], axis=1) # N-1
    dist = np.concatenate([[0], dist], axis=0)  # N
    dist = np.cumsum(dist).tolist()  # N

    if align:
        est_pose = AlignPose(est_pose, gt_pose).cpu().numpy()
    else:
        est_pose = est_pose.cpu().numpy()
    gt_pose = gt_pose.cpu().numpy()

    rot_errors, trans_errors = [], []
    
    for i in range(0, gt_pose.shape[0], 10):
        for length in subsequence_length:
            # find the end frame
            for j in range(i + 1, gt_pose.shape[0]):
                if dist[j] - dist[i] >= length:
                    break
            if j >= gt_pose.shape[0] - 1:
                break
            # compute the relative pose
            pose_delta_gt = np.linalg.inv(gt_pose[i]) @ gt_pose[j]
            pose_delta_est = np.linalg.inv(est_pose[i]) @ est_pose[j]
            # compute the error
            diff_pose = np.linalg.inv(pose_delta_est) @ pose_delta_gt
            rot_error = ((np.trace(diff_pose[:3, :3]) - 1) / 2).clip(-1, 1)
            rot_error = np.arccos(rot_error) * 180 / np.pi
            t_error = np.linalg.norm(diff_pose[:3, 3])
            
            rot_errors.append(rot_error/length)
            trans_errors.append(t_error/length)

    return {
        "mean_rot_error": np.mean(rot_errors),
        "median_rot_error": np.median(rot_errors),
        "max_rot_error": np.max(rot_errors),
        "min_rot_error": np.min(rot_errors),
        "rmse_rot_error": np.sqrt(np.mean(np.square(rot_errors))),
        "mean_trans_error": np.mean(trans_errors),
        "median_trans_error": np.median(trans_errors),
        "max_trans_error": np.max(trans_errors),
        "min_trans_error": np.min(trans_errors),
        "rmse_trans_error": np.sqrt(np.mean(np.square(trans_errors))),
    }

    

def camera_pose_visualize_coord_Rt(plyfile:str, R_wc_list, t_wc_list):
    if len(R_wc_list) == 0:
        print("no camera rotation")
        return False
    assert len(R_wc_list) == len(t_wc_list)
    cameras = []
    size = 0.1

    for i in range(len(R_wc_list)):
        vertex = []
        R_wc = R_wc_list[i]
        t_wc = t_wc_list[i]

        v1 = np.array([size, 0, 0])
        v2 = np.array([0, size, 0])
        v3 = np.array([0, 0, size])

        vertex.append(t_wc)
        vertex.append(np.dot(R_wc, v1) + t_wc)
        vertex.append(np.dot(R_wc, v2) + t_wc)
        vertex.append(np.dot(R_wc, v3) + t_wc)

        cameras.append(vertex)

    with open(plyfile, 'w') as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write("element vertex {}\n".format(len(cameras) * len(cameras[0])))
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write("element edge {}\n".format(len(cameras) * 3))
        fp.write("property int vertex1\n")
        fp.write("property int vertex2\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write("end_header\n")
        for i in range(len(cameras)):
            fp.write("{} {} {} 255 255 255\n".format(cameras[i][0][0], cameras[i][0][1], cameras[i][0][2]))
            fp.write("{} {} {} 255 0 0\n".format(cameras[i][1][0], cameras[i][1][1], cameras[i][1][2]))
            fp.write("{} {} {} 0 255 0\n".format(cameras[i][2][0], cameras[i][2][1], cameras[i][2][2]))
            fp.write("{} {} {} 0 0 255\n".format(cameras[i][3][0], cameras[i][3][1], cameras[i][3][2]))
        for i in range(len(cameras)):
            
            fp.write("{} {} 255 255 255\n".format(i*len(cameras[i]), i*len(cameras[i]) + 1))
            fp.write("{} {} 255 255 255\n".format(i*len(cameras[i]), i*len(cameras[i]) + 2))
            fp.write("{} {} 255 255 255\n".format(i*len(cameras[i]), i*len(cameras[i]) + 3))

    return True

# 理论上输入的应该是 T_wc 即从相机到世界的变换
# 如果输入的是世界到相机的变换，那么就要取逆，即 inverse=True
def camera_pose_visualize_coord(plyfile, T_wc_list:Tuple[torch.Tensor, np.ndarray], inverse=False):
    # N x 4 x 4
    # torch -> numpy 
    if(isinstance(T_wc_list, torch.Tensor)):
        T_wc_list = T_wc_list.detach().cpu().numpy()
    # numpy -> list
    if(isinstance(T_wc_list, np.ndarray)):
        T_wc_list = [T_wc_list[i] for i in range(T_wc_list.shape[0])]
    if inverse:
        T_wc_list = [np.linalg.inv(T_wc) for T_wc in T_wc_list]
    R_wc_list = [T[:3, :3] for T in T_wc_list]
    t_wc_list = [T[:3, 3] for T in T_wc_list]
    return camera_pose_visualize_coord_Rt(plyfile, R_wc_list, t_wc_list)

def camera_center_visualize_t(pcdfile, t_wc_list):
    '''
    用 pcd 文件可视化相机中心, 并且给每个点都赋予不同的intensity
    t_wc_list: list of camera center
    '''
    if len(t_wc_list) == 0:
        print("no camera center")
        return False
    with open(pcdfile, 'w') as fp:
        fp.write("VERSION .7\n")
        fp.write("FIELDS x y z intensity\n")
        fp.write("SIZE 4 4 4 4\n")
        fp.write("TYPE F F F F\n")
        fp.write("COUNT 1 1 1 1\n")
        fp.write("WIDTH {}\n".format(len(t_wc_list)))
        fp.write("HEIGHT 1\n")
        fp.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        fp.write("POINTS {}\n".format(len(t_wc_list)))
        fp.write("DATA ascii\n")
        for i in range(len(t_wc_list)):
            fp.write("{} {} {} {}\n".format(t_wc_list[i][0], t_wc_list[i][1], t_wc_list[i][2], i))
    return True

'''
Visualize camera centers
pcdfile: output pcd file name
T_wc_list: camera to world transformation matrix, [N, 4, 4]. It can be torch.Tensor or np.ndarray
inverse: if True, then T_wc_list is world to camera transformation matrix.
'''
def camera_center_visualize(pcdfile:str, T_wc_list:Union[torch.Tensor, np.ndarray], inverse=False):
    # N x 4 x 4
    # torch -> numpy 
    if(isinstance(T_wc_list, torch.Tensor)):
        T_wc_list = T_wc_list.detach().cpu().numpy()
    # numpy -> list
    if(isinstance(T_wc_list, np.ndarray)):
        T_wc_list = [T_wc_list[i] for i in range(T_wc_list.shape[0])]
    if inverse:
        T_wc_list = [np.linalg.inv(T_wc) for T_wc in T_wc_list]
    t_wc_list = [T[:3, 3] for T in T_wc_list]
    return camera_center_visualize_t(pcdfile, t_wc_list)




'''
Load the pose file in colmap format: 

IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
POINTS2D[] as (X, Y, POINT3D_ID)

return: Dict[name, pose]    name = file_name, pose = T_wc
'''

def loadColmapPose(path:str) -> Dict[str, torch.Tensor]:
    poses = {}
    first_line = 0
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                continue
            if len(poses) == 0:
                first_line = i
            # skip the POINTS2D line
            if (i - first_line) % 2 == 1:
                continue
            elems = line.strip().split()
            if len(elems) == 0:
                continue
            q_cw = torch.tensor([
                float(elems[2]),
                float(elems[3]),
                float(elems[4]),
                float(elems[1]),
            ])
            R_cw = roma.unitquat_to_rotmat(q_cw)
            t_cw = torch.tensor([float(elems[5]), float(elems[6]), float(elems[7])])
            T_cw = torch.eye(4)
            T_cw[:3, :3] = R_cw
            T_cw[:3, 3] = t_cw
            T_wc = torch.inverse(T_cw)

            name = elems[-1]
            poses[name] = T_wc
    return poses

'''
Load the pose file in KITTI format: 
name R_wc(0,0) R_wc(0,1) R_wc(0,2) t_wc(0) R_wc(1,0) R_wc(1,1) R_wc(1,2) t_wc(1) R_wc(2,0) R_wc(2,1) R_wc(2,2) t_wc(2)

return: Dict[name, pose]    name = file_name, pose = T_wc
'''
def loadKittiPose(pose_file:str) -> Dict[str, torch.Tensor]:
    poses = {}
    with open(pose_file, "r") as f:
        for line in f:
            elems = line.strip().split()
            if len(elems) == 0:
                continue
            try:
                name = elems[0].split("/")[-1]
                pose = [float(x) for x in elems[1:]]
                pose = torch.tensor(pose).reshape(3, 4)
                pose = torch.cat([pose, torch.tensor([[0, 0, 0, 1]])], dim=0)
                poses[name] = pose
            except:
                print(line)
    return poses

"""
est_pose: [N, 4, 4]   T_wc
gt_pose: [N, 4, 4]      T_wc
return: Dict[str, float]   key = error metric, value = error
"""
def evaluate_pose(est_pose:torch.Tensor, gt_pose:torch.Tensor) -> Dict[str, float]:
    print("evaluate pose with {} samples".format(len(gt_pose)))
    if(torch.allclose(est_pose, gt_pose, atol=1e-6)):
        rot_errors = [0 for _ in range(len(gt_pose))]
        trans_errors = [0 for _ in range(len(gt_pose))]
    else:
        # aligned_est_pose = AlignPoseRansac(est_pose, gt_pose)
        aligned_est_pose = AlignPose(est_pose, gt_pose)
        camera_center_visualize("pred_after_align.pcd", aligned_est_pose)
        camera_center_visualize("gt_after_align.pcd", gt_pose)
        est_rotations = aligned_est_pose[:, :3, :3].cpu().numpy()
        gt_rotations = gt_pose[:, :3, :3].cpu().numpy()

        rot_errors = [np.inf for _ in range(len(gt_rotations))]
        for _ in range(5):
            gt_rotations_aligned, est_rotations_aligned = AlignRotations(gt_rotations, est_rotations)
            errors = RotationError(gt_rotations_aligned, est_rotations_aligned)
            if(np.mean(errors) < np.mean(rot_errors)):
                rot_errors = errors

        est_centers = aligned_est_pose[:, :3, 3].cpu().numpy()  # N x 3
        gt_centers = gt_pose[:, :3, 3].cpu().numpy()  # N x 3
        trans_errors = np.sum((gt_centers - est_centers) ** 2, axis=1) ** 0.5

    return {
        "mean_rot_error": np.mean(rot_errors),
        "median_rot_error": np.median(rot_errors),
        "max_rot_error": np.max(rot_errors),
        "min_rot_error": np.min(rot_errors),
        "rmse_rot_error": np.sqrt(np.mean(np.square(rot_errors))),
        "rot_errors": rot_errors,
        "mean_trans_error": np.mean(trans_errors),
        "median_trans_error": np.median(trans_errors),
        "max_trans_error": np.max(trans_errors),
        "min_trans_error": np.min(trans_errors),
        "rmse_trans_error": np.sqrt(np.mean(np.square(trans_errors))),
    }



'''
convert image name to number, e.g. xxx/xxx/x_efg_123_xx.abc -> 123
'''
def name_to_num(name:str) -> int:
    name = os.path.basename(name)   # xxx/xxx/xxx.abc  -> xxx.abc
    name = name.split(".")[0]       # xxx.abc -> xxx
    start, end = 0, len(name)
    for i in range(len(name)-1, -1, -1):
        if name[i].isdigit():
            end = i
            break
    for i in range(end, -1, -1):
        if not name[i].isdigit():
            start = i
            break
    # the first character is digit
    if i == 0:
        return int(name[start:end+1])
    return int(name[start+1:end+1])


def main():
    gt_pose_path = "xxxx.txt"
    pred_pose_path = "xxxx.txt"
    gt_pose = loadKittiPose(gt_pose_path)
    pred_pose = loadKittiPose(pred_pose_path)

    same_key = set(gt_pose.keys()) & set(pred_pose.keys())
    # 把位姿按照文件名排序, 文件名是按照时间排序的, 所以位姿也会按照时间排序
    sorted_key = sorted(list(same_key), key=lambda x: name_to_num(x))
    print("Same key: ", len(sorted_key))

    gt_pose = [gt_pose[key] for key in sorted_key]
    pred_pose = [pred_pose[key] for key in sorted_key]

    gt_pose = torch.stack(gt_pose, dim=0)
    pred_pose = torch.stack(pred_pose, dim=0)
    eval_result = evaluate_pose(pred_pose, gt_pose)

    # 单独输出一下旋转误差
    rot_errors = eval_result.pop("rot_errors")
    rot_errors = [(sorted_key[i], rot_errors[i]) for i in range(len(rot_errors))]
    rot_errors = sorted(rot_errors, key=lambda x: x[1], reverse=True)
    print("Top 10 rot errors: ")
    for i in range(10):
        print(rot_errors[i])
    # 输出整体评估结果
    for key in eval_result:
        print("{}: {:.4f}".format(key, eval_result[key]))