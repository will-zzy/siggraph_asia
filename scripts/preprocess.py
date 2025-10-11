import argparse
import sys
from pathlib import Path
import shutil
from typing import Dict, Tuple, Optional, List, Set
import cv2
import re

# ========== 工具 ==========
INT_TS_RE = re.compile(r'(?<!\d)(\d{8,})(?!\d)')
MAT4_RE = re.compile(r'\bmat4x4\(')

def read_lines_strip(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n\r") for ln in f]

def find_video(path: Path, name: Optional[str], pattern: Optional[str]) -> Optional[Path]:
    if name:
        cand = path / name
        return cand if cand.exists() else None
    if pattern:
        hits = sorted(path.glob(pattern))
        return hits[0] if hits else None
    for ext in ("*.mp4","*.mov","*.avi","*.mkv","*.MP4"):
        hits = sorted(path.glob(ext))
        if hits:
            return hits[0]
    return None

def parse_video_info(videoinfo_path: Path) -> Tuple[Dict[int,int], Dict[int,int]]:
    """返回 (ts->fid, fid->ts)"""
    ts2fid: Dict[int,int] = {}
    fid2ts: Dict[int,int] = {}
    for ln in read_lines_strip(videoinfo_path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        try:
            frame_id = int(parts[0]); ts = int(parts[1])
        except ValueError:
            continue
        ts2fid[ts] = frame_id
        fid2ts[frame_id] = ts
    if not ts2fid:
        raise RuntimeError(f"Empty or invalid videoInfo at {videoinfo_path}")
    return ts2fid, fid2ts

def extract_filename_token(tokens: List[str]) -> Optional[str]:
    # 文件名在 mat4x4( 之前的最后一个 token；若无 mat4x4(，则找最后一个含图像扩展的 token
    fname_idx = None
    for i, tk in enumerate(tokens):
        if MAT4_RE.match(tk):
            fname_idx = i - 1 if i - 1 >= 0 else None
            break
    if fname_idx is None:
        for i in range(len(tokens)-1, -1, -1):
            if tokens[i].lower().endswith((".jpg",".jpeg",".png")):
                fname_idx = i
                break
    if fname_idx is not None and 0 <= fname_idx < len(tokens):
        return tokens[fname_idx]
    return None

def best_timestamp_from_line(line: str) -> Optional[int]:
    tokens = line.split()
    fn = extract_filename_token(tokens)
    if fn:
        m = INT_TS_RE.search(fn)
        if m:
            return int(m.group(1))
    ints_in_line = [int(m.group(1)) for m in INT_TS_RE.finditer(line)]
    if len(ints_in_line) == 1:
        return ints_in_line[0]
    return None

def map_ts_to_frame_id(ts_target: int, ts2fid: Dict[int,int], max_rel_diff: float = 1e-9) -> Tuple[int,int]:
    if ts_target in ts2fid:
        return ts2fid[ts_target], 0
    best_ts, best_d = None, None
    for ts in ts2fid.keys():
        d = abs(ts - ts_target)
        if best_d is None or d < best_d:
            best_d, best_ts = d, ts
    if best_ts is None:
        raise RuntimeError("videoInfo timestamp map is empty unexpectedly.")
    rel = best_d / max(1.0, abs(ts_target))
    if rel > max_rel_diff:
        print(f"[warn] large ts gap for {ts_target}: nearest {best_ts}, delta={best_d} (rel={rel:.2e})")
    return ts2fid[best_ts], best_d or 0

def parse_images_txt_needed(images_txt: Path) -> List[Tuple[int, str]]:
    """
    解析 inputs/slam/images.txt，返回 [(timestamp, filename_token_basename), ...]
    只保留能解析出时间戳且有有效文件名 token 的行。
    """
    res: List[Tuple[int,str]] = []
    for ln in read_lines_strip(images_txt):
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        tokens = ln.split()
        fname = extract_filename_token(tokens)
        if not fname:
            continue
        ts = best_timestamp_from_line(ln)
        if ts is None:
            continue
        res.append((ts, Path(fname).name))  # 仅用基名写到 images/
    if not res:
        raise RuntimeError(f"No usable entries parsed from {images_txt}")
    return res

def export_selected_frames(video_path: Path, out_dir: Path, fid2names: Dict[int, List[str]]) -> Tuple[int,int]:
    """
    顺序读取视频，只当帧号在 fid2names 时写出，对应到多个目标文件名。
    返回 (写出的图像数, 实际读取到的最后帧号)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    max_fid = max(fid2names.keys())
    idx = 1
    written = 0
    while idx <= max_fid:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in fid2names:
            for name in fid2names[idx]:
                cv2.imwrite(str(out_dir / name), frame)
                written += 1
        idx += 1
    cap.release()
    return written, idx - 1

def prune_images_dir(out_dir: Path, keep_names: Set[str]):
    if not out_dir.exists():
        return
    for p in out_dir.iterdir():
        if p.is_file() and p.name not in keep_names:
            try:
                p.unlink()
            except Exception as e:
                print(f"[warn] failed to remove {p}: {e}")

def copy_slam_texts(case_dir: Path):
    src_dir = case_dir / "inputs" / "slam"
    dst_dir = case_dir / "sparse" / "0"
    dst_dir.mkdir(parents=True, exist_ok=True)
    candidates = ["images.txt", "cameras.txt", "camera.txt", "points3D.txt", "points.txt"]
    copied = []
    for name in candidates:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    if copied:
        print(f"[ok] copied to {dst_dir}: {', '.join(copied)}")
    else:
        print(f"[warn] no SLAM text files found in {src_dir}")

# ========== 主流程 ==========
def process_case(case_dir: Path, args):
    print(f"\n=== Processing case: {case_dir} ===")
    images_txt = case_dir / "inputs" / "slam" / "images.txt"
    video_info_txt = case_dir / args.videoinfo_txt
    video_path = find_video(case_dir, args.video_name, args.video_name_pattern)
    out_images = case_dir / args.out_images_dir

    if not images_txt.exists():
        raise FileNotFoundError(f"Missing images.txt: {images_txt}")
    if not video_info_txt.exists():
        raise FileNotFoundError(f"Missing videoInfo.txt: {video_info_txt}")
    if not video_path or not video_path.exists():
        raise FileNotFoundError(f"Video not found under {case_dir} (use --video-name or --video-name-pattern)")

    # 1) 只取 images.txt 里需要的文件（时间戳 + 原文件名）
    wanted_entries = parse_images_txt_needed(images_txt)  # [(ts, name), ...]
    keep_names = {name for _, name in wanted_entries}

    # 2) ts -> frame_id (最近邻)，只为 images.txt 里那些 ts 建立映射
    ts2fid, _ = parse_video_info(video_info_txt)
    fid2names: Dict[int, List[str]] = {}
    for ts, name in wanted_entries:
        fid, _ = map_ts_to_frame_id(ts, ts2fid)
        fid2names.setdefault(fid, []).append(name)

    # 3) 只导出这些帧；写入的文件名与 images.txt 一致（扩展名保持）
    written, last_read = export_selected_frames(video_path, out_images, fid2names)
    print(f"[info] written {written} images to {out_images} (read up to frame {last_read})")

    # 4) 清理 images/ 中的其他文件，保证“只包含 images.txt 中的序列帧”
    prune_images_dir(out_images, keep_names)
    print(f"[ok] pruned {out_images} to exactly match images.txt ({len(keep_names)} files)")

    # 5) 文本原封不动复制到 sparse/0
    copy_slam_texts(case_dir)

def main():
    ap = argparse.ArgumentParser(
        description="Export ONLY frames listed in inputs/slam/images.txt; names match images.txt; copy SLAM texts to sparse/0 unchanged."
    )
    ap.add_argument("--root", type=str, required=True, help="根目录（单案例或多案例父目录）")
    ap.add_argument("--video-name", type=str, default=None, help="视频文件名（如 video.mp4）")
    ap.add_argument("--video-name-pattern", type=str, default=None, help="视频通配（如 *.mp4）")
    ap.add_argument("--videoinfo-txt", type=str, default="inputs/videoInfo.txt")
    ap.add_argument("--out-images-dir", type=str, default="images", help="输出图片目录（相对案例目录）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}"); sys.exit(1)

    if (root / "inputs" / "slam" / "images.txt").exists() and (root / args.videoinfo_txt).exists():
        case_dirs = [root]
    else:
        case_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and
                     (p / "inputs" / "slam" / "images.txt").exists() and
                     (p / args.videoinfo_txt).exists()]
    if not case_dirs:
        print("No valid case directories found."); sys.exit(1)

    for case in case_dirs:
        process_case(case, args)

if __name__ == "__main__":
    main()
