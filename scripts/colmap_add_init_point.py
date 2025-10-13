#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import cv2
import math
import re
from typing import Dict, Tuple, List

# =============== IO ===============
def read_lines(p: Path):
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_cameras(cameras_txt: Path) -> Dict[int, dict]:
    """
    读取 cameras.txt（假定 PINHOLE fx fy cx cy；忽略畸变）。
    返回 {CAMERA_ID: {"width":W,"height":H,"fx":...,"fy":...,"cx":...,"cy":...,"K":3x3}}
    支持行形如：
      0 PINHOLE 480 640 448.246002 448.579987 238.580994 320.403015
      或行尾多出畸变参数，忽略即可
    """
    cams = {}
    for ln in read_lines(cameras_txt):
        if ln.startswith("#"): 
            continue
        toks = ln.split()
        if len(toks) < 8:
            continue
        cam_id = int(toks[0])
        model = toks[1].upper()
        h = int(toks[2]); w = int(toks[3])
        fx, fy, cx, cy = map(float, toks[4:8])
        if model not in ("PINHOLE","OPENCV","OPENCV_FISHEYE","OPENCV_FISHEYE4","RADIAL","SIMPLE_PINHOLE"):
            # 兼容：当不是 PINHOLE 也先按前四个当 fx,fy,cx,cy 用
            pass
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        cams[cam_id] = dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy, K=K)
    if not cams:
        raise RuntimeError(f"No valid camera in {cameras_txt}")
    return cams

def q_to_R(qw,qx,qy,qz) -> np.ndarray:
    """ 四元数 -> 旋转矩阵（w,x,y,z） """
    q = np.array([qw,qx,qy,qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    return R

def load_images_and_poses(images_txt: Path) -> List[dict]:
    """
    读 images.txt：期望每行含
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID TIMESTAMP FILENAME ...
    返回列表（按文件中顺序）:
      dict(image_id, cam_id, fname, R, t)
    约定 R,t 均为 world-to-camera:  [R|t]
    """
    items = []
    for ln in read_lines(images_txt):
        if ln.startswith("#"):
            continue
        toks = ln.split()
        if len(toks) < 11:
            continue
        img_id = int(toks[0])
        qw,qx,qy,qz = map(float, toks[1:5])
        tx,ty,tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        # 第 9 位是 timestamp，第 10 位才是文件名；你之前样例是这样的
        # 如和你文件不一致，可把下面两行索引改成对应列
        timestamp = toks[9]
        # 文件名 token 可能在后面（mat4x4 之前的最后一个 jpg/png）
        fname = None
        for tk in reversed(toks):
            if tk.lower().endswith((".jpg",".jpeg",".png")):
                fname = tk
                break
        if fname is None:
            # 退化：就用 toks[10]
            fname = toks[10] if len(toks) > 10 else f"{img_id}.png"

        R_wc = q_to_R(qw,qx,qy,qz)        # 通常 images.txt 给的是 cam->world ? 你的样例给的是 W2C.R 的转置
        # 你前面的 Camera 类说：传进来的 R 是 W2C.R 的转置
        # 为避免歧义，我们这里采用最常见的：R = R_wc (world->cam), t = [tx,ty,tz] (world->cam)
        # 若结果方向反了，调试时可以替换为 R = R_wc.T, t = -R_wc.T @ t
        R = R_wc
        t = np.array([tx,ty,tz], dtype=np.float64).reshape(3,1)

        items.append(dict(image_id=img_id, cam_id=cam_id, fname=fname, R=R, t=t))
    if not items:
        raise RuntimeError(f"No valid images in {images_txt}")
    return items

# =============== Triangulation ===============
def build_P(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R, t])  # 3x4
    return K @ Rt

def sift_match(img1, img2):
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    if d1 is None or d2 is None: 
        return [], [], []
    index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])  # (N,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])  # (N,2)
    return good, pts1, pts2

def triangulate_pair(img1, img2, P1, P2, reproj_th=2.0):
    """
    输入像素坐标（未归一化）和投影矩阵 P1,P2，返回三角化后的 3D 点（已做可见性和重投影误差过滤）
    """
    if len(img1) < 8:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.uint8)

    # 齐次三角化
    Xh = cv2.triangulatePoints(P1, P2, img1.T, img2.T)  # 4xN
    X = (Xh[:3] / Xh[3:]).T                              # Nx3

    # 可见性过滤（深度为正）
    # 把点投到相机坐标：Z>0
    def cam_depth(R, t, X):
        # X_cam = R*X + t
        return (X @ R.T + t.ravel())[:, 2]
    # 后面会填
    return X

def reproject_and_filter(X, K1, R1, t1, K2, R2, t2, pts1, pts2, reproj_th=2.0):
    """
    对已三角化点做可视性与重投影过滤，返回保留后的点以及对应的颜色取样索引
    """
    # 可见性
    z1 = (X @ R1.T + t1.ravel())[:, 2]
    z2 = (X @ R2.T + t2.ravel())[:, 2]
    vis = (z1 > 0) & (z2 > 0)

    # 重投影
    def project(K,R,t,X):
        x = X @ R.T + t.ravel()
        x = (x / x[:,2:3])[:, :2]
        uv = (x @ np.array([[K[0,0], 0],[0, K[1,1]]])) + np.array([K[0,2], K[1,2]])
        return uv

    uv1 = project(K1,R1,t1,X)
    uv2 = project(K2,R2,t2,X)

    e1 = np.linalg.norm(uv1 - pts1, axis=1)
    e2 = np.linalg.norm(uv2 - pts2, axis=1)
    ok = vis & (e1 < reproj_th) & (e2 < reproj_th)
    return X[ok], ok

def sample_rgb(img, uv):
    """最近邻采样颜色"""
    h, w = img.shape[:2]
    u = np.clip(np.round(uv[:,0]).astype(int), 0, w-1)
    v = np.clip(np.round(uv[:,1]).astype(int), 0, h-1)
    col = img[v, u, :].reshape(-1,3)
    return col

# =============== Main ===============
def main():
    ap = argparse.ArgumentParser("Quick triangulation init (SIFT+triangulate) to build sparse/0/points3D.txt")
    ap.add_argument("--root", required=True, help="案例根目录")
    ap.add_argument("--images-txt", default="inputs/slam/images.txt")
    ap.add_argument("--cameras-txt", default="inputs/slam/cameras.txt")
    ap.add_argument("--images-dir",  default="images")
    ap.add_argument("--out-sparse",  default="sparse/0")
    ap.add_argument("--pair-stride", type=int, default=5, help="匹配 pair(i, i+stride)")
    ap.add_argument("--max-pairs", type=int, default=50, help="最多处理多少个 pair（快速验证）")
    ap.add_argument("--reproj-th", type=float, default=2.0)
    ap.add_argument("--voxel", type=float, default=0.01, help="体素下采样网格大小(米)")
    args = ap.parse_args()

    root = Path(args.root)
    images_txt = root / args.images_txt
    cameras_txt = root / args.cameras_txt
    images_dir = root / args.images_dir
    out_dir = root / args.out_sparse
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = load_cameras(cameras_txt)
    items = load_images_and_poses(images_txt)

    # 选择若干对（邻近/有基线），先用简单的 stride
    pairs = []
    for i in range(0, len(items)-args.pair_stride, args.pair_stride):
        pairs.append((i, i+args.pair_stride))
        if len(pairs) >= args.max_pairs:
            break
    if not pairs:
        print("No pairs selected, adjust --pair-stride / --max-pairs")
        return

    all_pts = []
    all_rgb = []

    for (i, j) in pairs:
        a = items[i]; b = items[j]
        Ka = cams[a["cam_id"]]["K"]; Kb = cams[b["cam_id"]]["K"]
        Pa = build_P(Ka, a["R"], a["t"])
        Pb = build_P(Kb, b["R"], b["t"])

        imgA = cv2.imread(str(images_dir / a["fname"]), cv2.IMREAD_COLOR)
        imgB = cv2.imread(str(images_dir / b["fname"]), cv2.IMREAD_COLOR)
        if imgA is None or imgB is None:
            print(f"[warn] missing image {a['fname']} or {b['fname']}, skip")
            continue

        # SIFT + ratio test
        matches, pts1, pts2 = sift_match(imgA, imgB)
        if len(matches) < 16:
            print(f"[pair] {a['fname']} vs {b['fname']} : matches={len(matches)} -> skip")
            continue

        Xh = cv2.triangulatePoints(Pa, Pb, pts1.T, pts2.T)
        X = (Xh[:3] / Xh[3:]).T  # Nx3

        # 过滤（正深度 + 重投影）
        X_ok, ok_mask = reproject_and_filter(
            X, Ka, a["R"], a["t"], Kb, b["R"], b["t"], pts1, pts2, reproj_th=args.reproj_th
        )
        if X_ok.shape[0] == 0:
            print(f"[pair] {a['fname']} vs {b['fname']} : no valid triangulated points")
            continue

        # 为了快速，颜色直接取第一张图的像素
        uv1_ok = pts1[ok_mask]
        col = sample_rgb(imgA, uv1_ok)[:, ::-1]  # BGR->RGB

        all_pts.append(X_ok)
        all_rgb.append(col.astype(np.uint8))

        print(f"[pair] {a['fname']} vs {b['fname']} : tri_pts={X_ok.shape[0]} (accum {sum(x.shape[0] for x in all_pts)})")

    if not all_pts:
        print("No triangulated points produced.")
        return

    P = np.concatenate(all_pts, axis=0)
    C = np.concatenate(all_rgb, axis=0)

    # 简单体素下采样（哈希到体素格）
    # if args.voxel > 0:
    #     vox = args.voxel
    #     keys = np.floor(P / vox).astype(np.int64)
    #     _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    #     P = P[uniq_idx]
    #     C = C[uniq_idx]

    # 写 points3D.txt（新建，ERROR 先置 1.0）
    def _get_last_point_id(txt_path: Path) -> int:
        if not txt_path.exists():
            return 0
        last_id = 0
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                try:
                    pid = int(parts[0])
                    last_id = max(last_id, pid)
                except Exception:
                    continue
        return last_id

    out_pts = out_dir / "points3D.txt"
    start_id = _get_last_point_id(out_pts)
    pid = start_id + 1
    with out_pts.open("a", encoding="utf-8") as f:
        for p, c in zip(P, C):
            r, g, b = int(c[0]), int(c[1]), int(c[2])
            f.write(f"{pid} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b} 1.000000\n")
            pid += 1

    print(f"[ok] wrote {P.shape[0]} points -> {out_pts}")

if __name__ == "__main__":
    main()
