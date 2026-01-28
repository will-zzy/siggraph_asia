

import torch
import numpy as np
import os
import json
from scene.gaussian_model import GaussianModel
import sys
from train_dash import eval
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from gaussian_renderer import render
from argparse import ArgumentParser, Namespace
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

        
if __name__ == "__main__":
    parser = ArgumentParser(description="transforming")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_file", type=str, default = None)
    parser.add_argument("--read_time_file", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(4, "sparse_adam")
    scene = Scene(dataset, gaussians, pipe, load_iteration=-1, shuffle=False)
    
    # Load global transform
    path = os.path.join(scene.model_path, "global_transform.txt")
    with open(path, "r") as f:
        lines = f.readlines()
    rows = [list(map(float, line.strip().split())) for line in lines]
    global_transform = torch.tensor(rows, dtype=torch.float32)
    
    # Load time info
    case_name = dataset.source_path.split("/")[-1]
    time_path=os.path.join(scene.model_path, args.read_time_file) # This is a json file
    with open(time_path, "r") as f:
        time_info = json.load(f)
    all_time = time_info[case_name]["time"]
    
    with torch.no_grad():
        eval(dataset, pipe, case_name, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), scene.loaded_iter, all_time, args.log_file, global_transform = global_transform)