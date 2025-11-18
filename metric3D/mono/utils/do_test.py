import torch
import torch.nn.functional as F
import logging
import os
import os.path as osp
from colmap_loader import *
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, create_html, save_raw_imgs, save_normal_val_imgs
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud
def read_colmap_model(path, ext=".txt"):
    cameras, images, points3D = read_model(path, ext)
    return cameras, images, points3D
def to_cuda(data: dict):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>=1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data

def align_scale(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    else:
        scale = 1
    pred_scaled = pred * scale
    return pred_scaled, scale

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    target_mask = target[mask].cpu().numpy()
    pred_mask = pred[mask].cpu().numpy()
    if torch.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale

def align_scale_shift_numpy(pred: np.array, target: np.array):
    mask = target > 0
    target_mask = target[mask]
    pred_mask = pred[mask]
    if np.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale


def build_camera_model(H : int, W : int, intrinsics : list) -> np.array:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map. 
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

    y_col = np.arange(0, H).astype(np.float32) 
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T # [H, W]

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y = np.arctan(y_center / (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model

# --- 稀疏投影生成 COLMAP 深度图 ----------------------------------------------
def project_points_to_depth(points3D, image, camera):
    R = image.qvec2rotmat()
    t = image.tvec
    K = np.zeros((3, 3), dtype=np.float32)
    fx, fy, cx, cy = camera.params[:4]
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1

    depth_map = np.zeros((camera.height, camera.width), dtype=np.float32)

    for p in points3D.values():
        X_world = p.xyz
        X_cam = R @ X_world + t
        if X_cam[2] <= 0:
            continue
        x_proj = K @ X_cam
        x_proj /= x_proj[2]
        u, v = int(x_proj[0]), int(x_proj[1])
        if 0 <= u < camera.width and 0 <= v < camera.height:
            d = X_cam[2]
            if depth_map[v, u] == 0 or depth_map[v, u] > d:
                depth_map[v, u] = d
    return depth_map

def resize_for_input(image, output_shape, intrinsic, canonical_shape, to_canonical_ratio):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    """
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # padding
    image = cv2.copyMakeBorder(
        image, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=padding)
    
    # Resize, adjust principle point
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio

    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(
        cam_model, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=-1)

    pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor=1/to_scale_ratio
    return image, cam_model, pad, label_scale_factor


def get_prediction(
    model: torch.nn.Module,
    input: torch.tensor,
    cam_model: torch.tensor,
    pad_info: torch.tensor,
    scale_info: torch.tensor,
    gt_depth: torch.tensor,
    normalize_scale: float,
    ori_shape: list=[],
):

    data = dict(
        input=input,
        cam_model=cam_model,
    )
    pred_depth, confidence, output_dict = model.module.inference(data)

    return pred_depth, output_dict

def transform_test_data_scalecano(rgb, intrinsic, data_basic):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = data_basic['canonical_space']
    forward_size = data_basic.crop_size
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # BGR to RGB
    #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space['focal_length']

    cano_label_scale_ratio = canonical_focal / ori_focal

    canonical_intrinsic = [
        intrinsic[0] * cano_label_scale_ratio,
        intrinsic[1] * cano_label_scale_ratio,
        intrinsic[2],
        intrinsic[3],
    ]

    # resize
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0)

    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb.cuda()
    
    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].cuda()
    cam_model_stacks = [
        torch.nn.functional.interpolate(cam_model, size=(cam_model.shape[2]//i, cam_model.shape[3]//i), mode='bilinear', align_corners=False)
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor

# def align_depths(pred, sparse):
#     mask = (sparse>0.01)
#     A = np.vstack([pred[mask], np.ones(mask.sum())]).T
#     y = sparse[mask]
#     a, b = np.linalg.lstsq(A, y, rcond=None)[0]
#     return a*pred + b
def align_depths(pred, sparse, max_iters=1000, min_inliers=20, inlier_thresh=None, rng_seed=0):
    """
    用 RANSAC 对齐单目深度 pred 到稀疏 COLMAP 深度 sparse，拟合 y ≈ a*pred + b。
    返回对齐后的深度：a*pred + b
    """
    pred = np.asarray(pred)
    sparse = np.asarray(sparse)

    # 有效掩码：稀疏深度>0 且两者都是有限值
    mask = (sparse > 0.01) & np.isfinite(sparse) & np.isfinite(pred)
    if mask.sum() < min_inliers:
        # 样本太少，保守返回
        return pred

    x = pred[mask].astype(np.float64)
    y = sparse[mask].astype(np.float64)

    # 残差阈值（绝对值）。若未给，按 y 的中位数给个相对阈值（5%）
    if inlier_thresh is None:
        med_y = np.median(y)
        # 用 robust range 防止极端值影响
        p5, p95 = np.percentile(y, [5, 95])
        span = max(1e-6, p95 - p5)
        # 取两者中更稳的一个尺度
        inlier_thresh = max(1e-3, min(0.05 * max(med_y, 1.0), 0.1 * span))

    # RANSAC
    rng = np.random.default_rng(rng_seed)
    N = x.shape[0]
    best_inliers = None
    best_count = -1
    best_a, best_b = 1.0, 0.0

    for _ in range(max_iters):
        # 随机采两点
        i1, i2 = rng.integers(0, N, size=2)
        if i1 == i2:
            continue
        x1, y1 = x[i1], y[i1]
        x2, y2 = x[i2], y[i2]
        denom = (x2 - x1)
        if abs(denom) < 1e-12:
            continue
        a = (y2 - y1) / denom
        b = y1 - a * x1

        # 计算残差并统计内点
        res = np.abs(a * x + b - y)
        inliers = res < inlier_thresh
        cnt = int(inliers.sum())

        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_a, best_b = a, b

    # 如果没有足够内点，直接返回
    if best_inliers is None or best_count < min_inliers:
        return pred

    # 用内点再做一次最小二乘拟合，得到最终 a,b
    x_in = x[best_inliers]
    y_in = y[best_inliers]
    if x_in.size >= 2:
        # np.polyfit 返回 [a, b]
        a_refit, b_refit = np.polyfit(x_in, y_in, deg=1)
        # 若出现负尺度，回退到中位数比例，偏移为0（常见 fallback）
        if a_refit < 0:
            a_refit = np.median(y_in) / (np.median(x_in) + 1e-8)
            b_refit = 0.0
        best_a, best_b = float(a_refit), float(b_refit)

    return best_a * pred + best_b

def depths_to_points(intrinsics, depthmap, H, W, factor=1.0):
    c2w = torch.eye(4, device = "cuda")
    # fx = W / (2 * math.tan(view.FoVx / 2.))
    # fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[intrinsics[0] * factor, 0., intrinsics[2] * factor],
        [0., intrinsics[1] * factor, intrinsics[3] * factor],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    # rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    # rays_o = c2w[:3, 3]
    
    
    rays_o = torch.zeros_like(c2w[:3, 3]) # 相机坐标系
    rays_d = points @ intrins.inverse().T
    
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points, rays_d

def depth_to_normal(view, depth, H, W, factor):
    depth = torch.tensor(depth).cuda()
    points, rays_d = depths_to_points(view, depth, H, W, factor)
    points = points.reshape(*depth.shape[0:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1) # 逆着的
    output[1:-1, 1:-1, :] = normal_map
    return output.detach().cpu().numpy(), points.detach().cpu().numpy()


def do_scalecano_test_with_custom_data(
    model: torch.nn.Module,
    cfg: dict,
    test_data: list,
    logger: logging.RootLogger,
    root_dir,
    is_distributed: bool = True,
    local_rank: int = 0,
    bs: int = 2,  # Batch size parameter
    vis=True,
):

    show_dir = cfg.show_dir
    output_dir = cfg.output_dir + "/mono_depths"
    os.makedirs(output_dir, exist_ok=True)
    save_interval = 1
    # save_imgs_dir = show_dir + '/vis'
    # os.makedirs(save_imgs_dir, exist_ok=True)
    # save_pcd_dir = show_dir + '/pcd'
    # os.makedirs(save_pcd_dir, exist_ok=True)
    train_test_split_path = os.path.join(show_dir, "train_test_split.json")
    with open(train_test_split_path, 'r') as f:
        train_test_exp = json.load(f)
    # normalize_scale = cfg.data_basic.depth_range[1]
    # dam = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
    # dam_median = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
    # dam_global = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
    
    # Process data in batches
    model_dir = show_dir + "/sparse/0"
    cameras, images, points3D = read_model(model_dir, ".txt")
    
    prefix = "/images"
    image_dir = show_dir + prefix
    if not os.path.exists(image_dir):
        prefix = "/images_gt_downsampled"
        image_dir = show_dir + prefix
    img_paths = []
    print("visualize mono results:", vis)
    pbar = tqdm(total=len(train_test_exp['train']), desc='Processing Images')
    for image_id, image in images.items():
        rgb_inputs, pads, label_scale_factors, gt_depths, rgb_origins = [], [], [], [], []
        img_path = os.path.join(image_dir, image.name)
        if not os.path.exists(img_path):
            continue
        if image.name in train_test_exp['test']:
            continue
        img_paths.append(img_path)
        cam = cameras[image.camera_id]
        intrinsic = cam.params[:4]  # fx, fy, cx, cy
        
        rgb_origin = cv2.imread(img_path)[:, :, ::-1].copy()
        rgb_origins.append(rgb_origin)
        rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)
        rgb_inputs.append(rgb_input)
        pads.append(pad)
        label_scale_factors.append(label_scale_factor)
        
        pred_depths, outputs = get_prediction(
            model=model,
            input=torch.stack(rgb_inputs),  # Stack inputs for batch processing
            cam_model=None,
            pad_info=pads,
            scale_info=None,
            gt_depth=None,
            normalize_scale=None,
        )   
         
        pred_depth = pred_depths.squeeze()  # Remove the channel dimension
        pred_depth = pred_depth[pad[0] : pred_depth.shape[0] - pad[1], pad[2] : pred_depth.shape[1] - pad[3]]
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], [rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear').squeeze() # to original size
        pred_depth = pred_depth.detach().cpu().numpy()
        colmap_depth = project_points_to_depth(points3D, image, cam)
        
        aligned_depth = align_depths(pred_depth, colmap_depth)
        
        
        normal_out = outputs['prediction_normal'].squeeze()
        pred_normal = normal_out[:3, :, :] # (3, H, W)
        H, W = pred_normal.shape[1:]
        pred_normal = pred_normal[ :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

        pred_normal = torch.nn.functional.interpolate(pred_normal[None, :], size=[rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear', align_corners=True).squeeze()
        pred_normal = pred_normal.permute(1,2,0).detach().cpu().numpy()
        # os.makedirs(os.path.dirname(img_path).replace(prefix, "mono_depths"), exist_ok=True)
        # os.makedirs(os.path.dirname(img_path).replace(prefix, "mono_normals"), exist_ok=True)
        # os.makedirs(os.path.dirname(img_path).replace(prefix, "mono_depths_vis"), exist_ok=True)
        # os.makedirs(os.path.dirname(img_path).replace(prefix, "mono_normals_vis"), exist_ok=True)
        
        depth_save_dir = output_dir
        # normal_save_dir = os.path.dirname(img_path).replace("images", "mono_normals")
        # depth_vis_save_dir = os.path.dirname(img_path).replace("images", "mono_depths_vis")
        # normal_vis_save_dir = os.path.dirname(img_path).replace("images", "mono_normals_vis")
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        depth_save_path = os.path.join(depth_save_dir, base_name + "_depth.npy")
        # normal_save_path = os.path.join(normal_save_dir, base_name + "_normal.npy")
        # depth_vis_save_path = os.path.join(depth_vis_save_dir, base_name + ".png")
        # normal_vis_save_path = os.path.join(normal_vis_save_dir, base_name + ".png")

        # aligned_depth = aligned_depth.clamp(min=0.01, max=10000)
        aligned_depth = np.clip(aligned_depth, 0.01, 10000)
        inv_d = 1.0 / aligned_depth
        
        np.save(depth_save_path, inv_d) # 存成逆深度
        # np.save(depth_save_path, aligned_depth)
        # np.save(normal_save_path, pred_normal)
        # if vis:
            # cv2.imwrite(depth_vis_save_path, (aligned_depth / aligned_depth.max() * 255).astype(np.uint8))
            # cv2.imwrite(normal_vis_save_path, ((pred_normal + 1) / 2 * 255).astype(np.uint8))
        
        # normal_map, _ = depth_to_normal(intrinsic, aligned_depth, aligned_depth.shape[0], aligned_depth.shape[1], 1)
        # normal_map = cv2.resize(normal_map, (W, H), interpolation=cv2.INTER_NEAREST)
        pbar.update(1)
        
        
        
        

def postprocess_per_image(i, pred_depth, gt_depth, intrinsic, rgb_origin, normal_out, pad, an, dam, dam_median, dam_global, is_distributed, save_imgs_dir, save_pcd_dir, normalize_scale, scale_info):

    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad[0] : pred_depth.shape[0] - pad[1], pad[2] : pred_depth.shape[1] - pad[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], [rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear').squeeze() # to original size
    # pred_depth = pred_depth * normalize_scale / scale_info

    
    
    
    
    
    
    
    
    
    
    
    
    pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth
    if gt_depth is not None:

        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear').squeeze() # to original size

        gt_depth = torch.from_numpy(gt_depth).cuda()

        pred_depth_median = pred_depth * gt_depth[gt_depth != 0].median() / pred_depth[gt_depth != 0].median()
        pred_global, _ = align_scale_shift(pred_depth, gt_depth)
        
        mask = (gt_depth > 1e-8)
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, is_distributed)
        dam_median.update_metrics_gpu(pred_depth_median, gt_depth, mask, is_distributed)
        dam_global.update_metrics_gpu(pred_global, gt_depth, mask, is_distributed)
        print(gt_depth[gt_depth != 0].median() / pred_depth[gt_depth != 0].median(), )
    
    os.makedirs(osp.join(save_imgs_dir, an['folder']), exist_ok=True)
    rgb_torch = torch.from_numpy(rgb_origin).to(pred_depth.device).permute(2, 0, 1)
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None].to(rgb_torch.device)
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None].to(rgb_torch.device)
    rgb_torch = torch.div((rgb_torch - mean), std)

    save_val_imgs(
        i,
        pred_depth,
        gt_depth if gt_depth is not None else torch.ones_like(pred_depth, device=pred_depth.device),
        rgb_torch,
        osp.join(an['folder'], an['filename']),
        save_imgs_dir,
    )
    #save_raw_imgs(pred_depth.detach().cpu().numpy(), rgb_torch, osp.join(an['folder'], an['filename']), save_imgs_dir, 1000.0)

    # pcd
    pred_depth = pred_depth.detach().cpu().numpy()
    #pcd = reconstruct_pcd(pred_depth, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
    #os.makedirs(osp.join(save_pcd_dir, an['folder']), exist_ok=True)
    #save_point_cloud(pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3), osp.join(save_pcd_dir, an['folder'], an['filename'][:-4]+'.ply'))

    if an['intrinsic'] == None:
        #for r in [0.9, 1.0, 1.1]:
        for r in [1.0]:
            #for f in [600, 800, 1000, 1250, 1500]:
            for f in [1000]:
                pcd = reconstruct_pcd(pred_depth, f * r, f * (2-r), intrinsic[2], intrinsic[3])
                fstr = '_fx_' + str(int(f * r)) + '_fy_' + str(int(f * (2-r)))
                os.makedirs(osp.join(save_pcd_dir, an['folder']), exist_ok=True)
                save_point_cloud(pcd.reshape((-1, 3)), rgb_origin.reshape(-1, 3), osp.join(save_pcd_dir, an['folder'], an['filename'][:-4] + fstr +'.ply'))

    if normal_out is not None:
        pred_normal = normal_out[:3, :, :] # (3, H, W)
        H, W = pred_normal.shape[1:]
        pred_normal = pred_normal[ :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

        pred_normal = torch.nn.functional.interpolate(pred_normal[None, :], size=[rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear', align_corners=True).squeeze()

        gt_normal = None
        #if gt_normal_flag:
        if False:
            pred_normal = torch.nn.functional.interpolate(pred_normal, size=gt_normal.shape[2:], mode='bilinear', align_corners=True)    
            gt_normal = cv2.imread(norm_path)
            gt_normal = cv2.cvtColor(gt_normal, cv2.COLOR_BGR2RGB) 
            gt_normal = np.array(gt_normal).astype(np.uint8)
            gt_normal = ((gt_normal.astype(np.float32) / 255.0) * 2.0) - 1.0
            norm_valid_mask = (np.linalg.norm(gt_normal, axis=2, keepdims=True) > 0.5)
            gt_normal = gt_normal * norm_valid_mask               
            gt_normal_mask = ~torch.all(gt_normal == 0, dim=1, keepdim=True)
            dam.update_normal_metrics_gpu(pred_normal, gt_normal, gt_normal_mask, cfg.distributed)# save valiad normal

        save_normal_val_imgs(iter, 
                            pred_normal, 
                            gt_normal if gt_normal is not None else torch.ones_like(pred_normal, device=pred_normal.device),
                            rgb_torch, # data['input'], 
                            osp.join(an['folder'], 'normal_'+an['filename']), 
                            save_imgs_dir,
                            )

