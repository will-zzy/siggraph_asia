#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import struct
import math
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from mono.utils.do_test import get_prediction, transform_test_data_scalecano, project_points_to_depth, align_depths
import numpy as np
import cv2
from colmap_loader import *
import torch
from mono.utils.running import load_ckpt
from mono.model.monodepth_model import get_configured_monodepth_model
from tqdm import tqdm

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
# ---------------------------
# Utilities
# ---------------------------

def read_lines_strip(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n\r") for ln in f]

def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)

def load_cameras_txt(p: Path) -> Dict[int, Dict]:
    cams = {}
    for ln in read_lines_strip(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        toks = ln.split()
        # COLMAP cameras.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
        cam_id = int(toks[0])
        model = toks[1]
        w, h = int(toks[2]), int(toks[3])
        params = list(map(float, toks[4:]))
        # 解析 fx, fy, cx, cy（支持常见模型）
        fx = fy = cx = cy = None
        if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "RADIAL", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL_FISHEYE"):
            # 常见顺序：PINHOLE: fx, fy, cx, cy
            # SIMPLE_PINHOLE: f, cx, cy
            if model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params[0], params[1], params[2]
                fx, fy = f, f
            elif model in ("OPENCV", "FULL_OPENCV", "SIMPLE_RADIAL", "RADIAL"):
                # 前四个仍是 fx, fy, cx, cy，后面是畸变参数
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                # 其他：尽量按 fx,fy,cx,cy 取，拿不到则降级为简单猜测
                if len(params) >= 4:
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                elif len(params) >= 3:
                    f, cx, cy = params[0], params[1], params[2]
                    fx, fy = f, f
                else:
                    raise RuntimeError(f"Unsupported camera model/params in {p}: {model} {params}")
        else:
            # 尝试前四个
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                fx, fy = f, f
            else:
                raise RuntimeError(f"Unknown camera model: {model} with too few params")

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        cams[cam_id] = dict(model=model, width=w, height=h, params=params, K=K)
    if not cams:
        raise RuntimeError(f"No cameras parsed from {p}")
    return cams

def load_cameras_bin(p: Path) -> Dict[int, Dict]:
    # 简单读取 COLMAP cameras.bin（仅解析必要项）
    cams = {}
    with p.open("rb") as f:
        def read_uint64(): return struct.unpack("<Q", f.read(8))[0]
        def read_uint32(): return struct.unpack("<I", f.read(4))[0]
        def read_double(): return struct.unpack("<d", f.read(8))[0]
        num_cams = read_uint64()
        for _ in range(num_cams):
            cam_id = read_uint32()
            model_id = read_uint32()
            # 映射常见模型 id（COLMAP 定义），这里只做最常见的
            model_map = {
                0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
                4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
                8: "THIN_PRISM_FISHEYE", 9: "RADIAL_FISHEYE"
            }
            model = model_map.get(model_id, f"MODEL_{model_id}")
            width = read_uint64()
            height = read_uint64()
            num_params = read_uint64()
            params = [read_double() for _ in range(num_params)]
            # 取 fx, fy, cx, cy
            if model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params[0], params[1], params[2]
                fx, fy = f, f
            else:
                if len(params) >= 4:
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                elif len(params) >= 3:
                    f_, cx, cy = params[0], params[1], params[2]
                    fx, fy = f_, f_
                else:
                    raise RuntimeError(f"Unsupported camera model in bin: {model} {params}")
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            cams[cam_id] = dict(model=model, width=width, height=height, params=params, K=K)
    if not cams:
        raise RuntimeError(f"No cameras parsed from {p}")
    return cams

def load_images_txt(p: Path) -> List[Dict]:
    """
    解析 COLMAP images.txt（只用第一行：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME）
    返回每张图的：{id, q=[qw,qx,qy,qz], t=[tx,ty,tz], cam_id, name}
    """
    out = []
    lines = read_lines_strip(p)
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln or ln.startswith("#"): continue
        toks = ln.split()
        if len(toks) < 9:
            continue
        img_id = int(toks[0])
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        name = toks[9] if len(toks) >= 10 else f"{img_id}.png"
        out.append(dict(id=img_id, q=np.array([qw,qx,qy,qz]), t=np.array([tx,ty,tz]), cam_id=cam_id, name=name))
        # 跳过紧随其后的 2D-3D 匹配行（若存在）
        if i < len(lines) and lines[i] and not lines[i].startswith("#"):
            # 有些文件会有一行或空行，这里不强制跳，保持 i 不动亦可
            pass
    if not out:
        raise RuntimeError(f"No images parsed from {p}")
    return out

def get_camera_center_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # w2c: X_c = R X_w + t
    # center C_w 满足 R C_w + t = 0 => C_w = -R^T t
    return (-R.T @ t).reshape(3)

def build_w2c(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = qvec2rotmat(q)
    Rt = np.eye(4)
    Rt[:3,:3] = R
    Rt[:3, 3] = t
    return Rt

def inv_SE3(M: np.ndarray) -> np.ndarray:
    R = M[:3,:3]; t = M[:3,3]
    Minv = np.eye(4)
    Minv[:3,:3] = R.T
    Minv[:3,3] = -R.T @ t
    return Minv

def farthest_point_subset(centers: np.ndarray, k: int, image_ids) -> List[int]:
    """对相机中心做最远点采样，返回所选下标。"""
    n = centers.shape[0]
    if k >= n:
        return list(range(n))
    # 以几何中心为起点，先选距离均值最远的一个
    mean_c = centers.mean(axis=0, keepdims=True)
    d0 = np.linalg.norm(centers - mean_c, axis=1)
    sel = [int(np.argmax(d0))]
    fin_sel = [image_ids[int(np.argmax(d0))]]
    dist = np.linalg.norm(centers - centers[sel[0]], axis=1)
    for _ in range(1, k):
        # 到已选集合的“最近距离”
        for j in range(n):
            dist[j] = min(dist[j], np.linalg.norm(centers[j] - centers[sel[-1]]))
        sel.append(int(np.argmax(dist)))
        fin_sel.append(image_ids[int(np.argmax(dist))])
        
    return fin_sel

def read_points3D_max_id(p: Path) -> int:
    if not p.exists(): return 0
    max_id = 0
    for ln in read_lines_strip(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        try:
            pid = int(ln.split()[0])
            max_id = max(max_id, pid)
        except Exception:
            pass
    return max_id

def append_points3D(p: Path, rows: List[Tuple[int,float,float,float,int,int,int,float]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]} {r[5]} {r[6]} {r[7]:.6f}\n")


# ---------------------------
# Core pipeline
# ---------------------------
def sample_pixels(
    h: int, w: int, n: int, strategy: str = "uniform",
    depth: Optional[np.ndarray] = None,
    bins: int = 8,
) -> np.ndarray:
    """
    返回 (n,2) 整型像素坐标 (v,u)。

    strategy:
        - "uniform"      : 完全随机
        - "grid"         : 规则网格 + 抖动
        - "bg"           : 依据 depth 大值优先
        - "depth_uniform": 根据深度由近到远均匀采样
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    # ---------------- grid ----------------
    if strategy == "grid":
        side = int(math.sqrt(n))
        side = max(1, side)
        vv = np.linspace(0.5, h - 0.5, side)
        uu = np.linspace(0.5, w - 0.5, side)
        grid = np.stack(np.meshgrid(vv, uu, indexing="ij"), axis=-1).reshape(-1, 2)
        if grid.shape[0] > n:
            grid = grid[:n]
        jitter = np.random.uniform(-0.4, 0.4, size=grid.shape)
        pts = grid + jitter
        pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
        return np.round(pts).astype(np.int32)
    if strategy == "random":
        # 简单随机采样，均匀覆盖全图（不依赖深度）
        v = np.random.randint(0, h, size=(n,), dtype=np.int32)
        u = np.random.randint(0, w, size=(n,), dtype=np.int32)
        return np.stack([v, u], axis=-1)
    # ---------------- bg ----------------
    if strategy == "bg" and depth is not None:
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            thresh = np.nanpercentile(depth[valid], 80)
            mask = valid & (depth >= thresh)
            ys, xs = np.where(mask)
            if ys.size > 0:
                sel = np.random.choice(ys.size, size=min(n, ys.size), replace=False)
                return np.stack([ys[sel], xs[sel]], axis=-1).astype(np.int32)

    # ---------------- depth_uniform ----------------
    if strategy == "depth_uniform" and depth is not None:
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            return np.zeros((0, 2), dtype=np.int32)

        dvals = depth[valid]
        ys, xs = np.where(valid)

        # 均匀分成若干深度段
        d_min, d_max = np.nanmin(dvals), np.nanmax(dvals)
        bins = min(bins, n)  # 防止采样太稀
        edges = np.linspace(d_min, d_max, bins + 1)

        pts = []
        per_bin = max(1, n // bins)
        for i in range(bins):
            mask_bin = (dvals >= edges[i]) & (dvals < edges[i + 1])
            if not np.any(mask_bin):
                continue
            yb, xb = ys[mask_bin], xs[mask_bin]
            count = min(per_bin, len(yb))
            sel = np.random.choice(len(yb), size=count, replace=False)
            pts.append(np.stack([yb[sel], xb[sel]], axis=-1))
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        pts = np.concatenate(pts, axis=0)
        # 如果不够 n，再补一点随机
        if pts.shape[0] < n:
            extra = n - pts.shape[0]
            v = np.random.randint(0, h, size=extra, dtype=np.int32)
            u = np.random.randint(0, w, size=extra, dtype=np.int32)
            pts = np.concatenate([pts, np.stack([v, u], axis=-1)], axis=0)
        return pts[:n]

    # ---------------- uniform ----------------
    v = np.random.randint(0, h, size=(n,), dtype=np.int32)
    u = np.random.randint(0, w, size=(n,), dtype=np.int32)
    return np.stack([v, u], axis=-1)

def unproject_depth_to_world(uv: np.ndarray, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    uv: (n,2) 像素 (v,u)
    depth: (h,w) 深度
    K: 3x3
    c2w: 4x4
    返回 (n,3) 世界坐标
    """
    v = uv[:,0].astype(np.int32)
    u = uv[:,1].astype(np.int32)
    z = depth[v, u].astype(np.float64)
    # 过滤非正深度
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return np.zeros((0,3), dtype=np.float64)
    v, u, z = v[valid], u[valid], z[valid]
    # 像素 -> 相机坐标
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_c = np.stack([x, y, z, np.ones_like(z)], axis=0)  # 4×n
    pts_w = (c2w @ pts_c).T[:, :3]
    return pts_w

def run(
        cfg, 
        root: Path,
        images_dir: str = "images",
        depths_dir: str = "mono_depths",
        sparse_dir: str = "sparse/0",
        k_views: int = 8,
        samples_per_view: int = 2000,
        strategy: str = "bg",
        bg_percentile: float = 0.85,
        depth_scale: float = 1.0,
        min_depth: float = 1e-6,
        max_depth: float = 1e6,
        seed: int = 0, 
        model = None):

    np.random.seed(seed)
    random.seed(seed)

    sp_dir = root / sparse_dir
    cams_txt = sp_dir / "cameras.txt"
    cams_bin = sp_dir / "cameras.bin"
    imgs_txt = sp_dir / "images.txt"
    pts3d_txt = sp_dir / "points3D.txt"

    if not imgs_txt.exists():
        raise FileNotFoundError(f"Missing {imgs_txt}")
    if not cams_txt.exists() and not cams_bin.exists():
        raise FileNotFoundError(f"Missing cameras.txt/bin under {sp_dir}")

    cameras, images, points3D = read_model(sp_dir, ".txt")
    # load cameras
    # if cams_txt.exists():
    #     cams = load_cameras_txt(cams_txt)
    # else:
    #     cams = load_cameras_bin(cams_bin)

    # load images poses
    # images = load_images_txt(imgs_txt)

    # compute camera centers & select top-k by FPS
    centers = []
    pose_c2w = []
    image_ids = []
    for image_id, image in tqdm(images.items()):
        w2c = build_w2c(image.qvec, image.tvec)        # 4x4
        c2w = inv_SE3(w2c)
        pose_c2w.append(c2w)
        centers.append(c2w[:3, 3])
        image_ids.append(image_id)
    centers = np.stack(centers, axis=0)
    sel_idx = farthest_point_subset(centers, k_views, image_ids)

    # prepare id start
    start_id = read_points3D_max_id(pts3d_txt) + 1
    append_rows = []

    img_root = root / images_dir
    dep_root = root / depths_dir
    # model = get_configured_monodepth_model(cfg, )
    # model = torch.nn.DataParallel(model).cuda()
    # model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    for j, i in tqdm(enumerate(sel_idx)):
        im = images[i]
        cam = cameras[im.camera_id]
        intrinsic = cam.params[:4]
        w2c = build_w2c(im.qvec, im.tvec) 
        c2w = inv_SE3(w2c)
        
        
        K = np.array([
            [intrinsic[0], 0.0, intrinsic[2]],
            [0.0, intrinsic[1], intrinsic[3]],
            [0.0, 0.0, 1.0],
        ])
        # c2w = pose_c2w[i]

        # load image & depth
        
        rgb_inputs, pads = [], []
        img_path = img_root / im.name
        

        # img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
        rgb_origin = cv2.imread(img_path)[:, :, ::-1].copy()
        rgb_origin = cv2.resize(rgb_origin, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)
        
        rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)
        rgb_inputs.append(rgb_input)
        pads.append(pad)
        
        
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
        colmap_depth = project_points_to_depth(points3D, im, cam)
        
        aligned_depth = align_depths(pred_depth, colmap_depth)
        # aligned_depth = pred_depth
        
        # depth_vis_save_dir = os.path.dirname(img_path).replace("images", "mono_depths_vis")
        # os.makedirs(depth_vis_save_dir, exist_ok=True)
        # base_name = os.path.splitext(os.path.basename(img_path))[0]
        # depth_vis_save_path = os.path.join(depth_vis_save_dir, base_name + ".png")
        
        # os.makedirs(os.path.dirname(img_path).replace("images", "mono_depths_vis"), exist_ok=True)
        # # os.makedirs(os.path.dirname(img_path).replace("images", "mono_normals_vis"), exist_ok=True)
        # cv2.imwrite(depth_vis_save_path, (aligned_depth / aligned_depth.max() * 255).astype(np.uint8))
        
        normal_out = outputs['prediction_normal'].squeeze()
        pred_normal = normal_out[:3, :, :] # (3, H, W)
        H, W = pred_normal.shape[1:]
        pred_normal = torch.nn.functional.interpolate(pred_normal[None, :], size=[rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear', align_corners=True).squeeze()
        pred_normal = pred_normal.permute(1,2,0).detach().cpu().numpy()
        
        

        h, w = aligned_depth.shape[:2]

        depth = aligned_depth
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        depth = np.clip(depth, min_depth, max_depth)

        # sample pixels
        uv = sample_pixels(h, w, samples_per_view, strategy=strategy, depth=depth)
        if uv.shape[0] == 0:
            continue

        # back-project
        pts_w = unproject_depth_to_world(uv, depth, K, c2w)
        if pts_w.shape[0] == 0:
            continue

        # gather colors
        vv = uv[:pts_w.shape[0], 0]
        uu = uv[:pts_w.shape[0], 1]
        # BGR -> RGB
        cols = rgb_origin[vv, uu, ::-1]  # (n,3) uint8

        # append rows
        for k in range(pts_w.shape[0]):
            X, Y, Z = pts_w[k]
            R, G, B = map(int, cols[k])
            append_rows.append((start_id, float(X), float(Y), float(Z), R, G, B, 1.0))
            start_id += 1

        print(f"[{j+1}/{len(sel_idx)}] {im.name}: +{pts_w.shape[0]} pts")

    # write/append
    append_points3D(pts3d_txt, append_rows)
    print(f"[OK] Appended {len(append_rows)} points to {pts3d_txt}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser("Augment COLMAP points3D by back-projecting mono depths from top-k views")
    ap.add_argument('config', help='train config file path')
    ap.add_argument("--root", required=True, type=str, help="案例根目录（包含 images/, mono_depths/, sparse/0/）")
    ap.add_argument("--load-from", required=True, type=str, help="ckpt path")
    ap.add_argument("--images-dir", default="images", type=str, help="图像目录（相对 root）")
    ap.add_argument("--depths-dir", default="mono_depths", type=str, help="深度目录（相对 root，npy 同名文件）")
    ap.add_argument("--sparse-dir", default="sparse/0", type=str, help="COLMAP 稀疏目录（含 cameras.txt/.bin, images.txt, points3D.txt）")
    ap.add_argument("--k-views", default=8, type=int, help="选取用于增广的视角数（最远点采样）")
    ap.add_argument("--samples-per-view", default=2000, type=int, help="每个视角采样像素数")
    ap.add_argument("--strategy", default="bg", choices=["bg","uniform","grid", "random", "depth_uniform"], help="像素采样策略")
    ap.add_argument("--bg-percentile", default=0.85, type=float, help="背景偏置：选择深度分位（0~1）以上的像素")
    ap.add_argument("--depth-scale", default=1.0, type=float, help="深度缩放（你的单目深度若需要线性缩放）")
    ap.add_argument("--min-depth", default=1e-6, type=float)
    ap.add_argument("--max-depth", default=1e6, type=float)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    
    
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    model = get_configured_monodepth_model(cfg, )
    model = torch.nn.DataParallel(model).cuda()
    model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()
    run(cfg=cfg,
        root=Path(args.root),
        images_dir=args.images_dir,
        depths_dir=args.depths_dir,
        sparse_dir=args.sparse_dir,
        k_views=args.k_views,
        samples_per_view=args.samples_per_view,
        strategy=args.strategy,
        bg_percentile=args.bg_percentile,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        seed=args.seed,
        model=model)

if __name__ == "__main__":
    main()
