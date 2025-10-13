import glob
import os
import json
import cv2

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None
        normal = anno['normal'] if 'normal' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
            'normal': normal
        }
        datas.append(data_i)
    return datas

def load_data(path: str):
    rgbs = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True) + \
           glob.glob(os.path.join(path, '**', '*.png'), recursive=True)
    # intrinsic =  [1198.6254317093665, 1198.4757478852844, 800, 599.5] # GS
    intrinsic =  [3707.2463329009042, 3707.0677577927072, 2409.21, 1813.6900000000001] # E
    # intrinsic =  [1754.914032584707, 1754.7738321436555, 973.72500000000002, 538.86699999999996] # M
    # intrinsic =  [1814.049821337692, 1813.9942952718327, 973.72489896057095, 538.86675579718008] # S
    
    data = []
    for img_path in rgbs:
        filename = os.path.basename(img_path)  # 文件名，如 D.png

        # 计算 folder: 相对路径（相对于path的父目录）去掉文件名
        folder = os.path.relpath(os.path.dirname(img_path), start=path)
        

        data.append({
            'rgb': img_path,
            'depth': None,
            'intrinsic': intrinsic,
            'filename': filename,
            'folder': folder
        })

    return data