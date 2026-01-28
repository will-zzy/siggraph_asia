import os
import sys
import numpy as np
import torch
import time
import json
from torch import nn
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.camera_utils import update_pose, update_pose_by_global
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.schedule_utils import TrainingScheduler
from utils.fast_utils import sampling_cameras, compute_gaussian_score_fastgs
from utils.anysplat_utils import anySplat
import cv2

lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
    
    


def training(dataset, opt, pipe, debug_from, log_file=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    
    # Feed-forward Gaussains from AnySplat
    FF_gaussians, pred_all_extrinsic, pred_all_intrinsic = anySplat(dataset, opt, pipe)
    FF_gaussians.xyz_gradient_accum = torch.zeros((FF_gaussians.get_xyz.shape[0], 1), device=FF_gaussians.get_xyz.device)
    FF_gaussians.denom = torch.zeros((FF_gaussians.get_xyz.shape[0], 1), device=FF_gaussians.get_xyz.device)

    # Init Scene and Gaussains
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, pipe, FF_gaussians, shuffle=False)
    gaussians = scene.gaussians
    del FF_gaussians
    torch.cuda.empty_cache()
    
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Init DashGaussian scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians, 
                                  [cam.original_image for cam in scene.getTrainCameras()])

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1    
    
    data_device = torch.device("cuda")

    # Global pose delta
    cam_rot_delta = nn.Parameter(
        torch.zeros(3, requires_grad=True, device=data_device)
    )
    cam_trans_delta = nn.Parameter(
        torch.zeros(3, requires_grad=True, device=data_device)
    )
    global_transform = torch.eye(4, device=data_device)
    with open(os.path.join(scene.model_path, "global_transform.txt"), "w+") as f:
        for row in global_transform.cpu().numpy():
            f.write(" ".join([f"{x:.8f}" for x in row]) + "\n")
    l = [ 
        {'params': [cam_rot_delta], 'lr': 0.00002, "name": "pose_rot_delta"}, # 0.00008
        {'params': [cam_trans_delta], 'lr': 0.00001, "name": "pose_trans_delta"}, # 0.00005
    ]
    
    pose_optimizer = torch.optim.Adam(l)
    
    render_scale = scheduler.get_res_scale(1)
    start_time = time.time()
    
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Rescale GT image for DashGaussian
        gt_image = viewpoint_cam.original_image.cuda()
        if render_scale > 1:
            gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", recompute_scale_factor=True, antialias=True)[0]

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_image.shape[-2:], cam_rot_delta=cam_rot_delta, cam_trans_delta=cam_trans_delta)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            
            if render_scale > 1:
                mono_invdepth = torch.nn.functional.interpolate(mono_invdepth[None], scale_factor=1/render_scale, mode="bilinear", recompute_scale_factor=True, antialias=True)[0, 0]  

            Ll1depth_pure = torch.abs(invDepth  - mono_invdepth).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure * 0.3
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()        
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "Ll1depth": f"{ema_Ll1depth_for_log:.{4}f}", "N_GS": f"{gaussians._scaling.shape[0]}", "N_MAX": f"{scheduler.max_n_gaussian}", "R": f"{render_scale}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            # Check elapsed time
            elapsed_time = time.time() - start_time
            case_name = dataset.source_path.split('/')[-1]
            if elapsed_time >= 60 or iteration==opt.iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians at 60 seconds")
                all_time = elapsed_time
                eval(dataset, pipe, case_name, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), iteration, 60, log_file, global_transform)
                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}/point_cloud.ply".format(iteration))
                scene.gaussians.save_ply(f"{point_cloud_path}")                
                sys.exit(0)
            
            # Densification and pruning
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # Apply DashGaussian primitive scheduler to control densification.
                    densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_xyz.shape[0], render_scale)
                    
                    # Use fastgs strategy for densifing and pruning score computation
                    sample_view_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(sample_view_stack)
                    
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, dataset, opt, pipe, bg, DENSIFY=True, cam_rot_delta=cam_rot_delta, cam_trans_delta=cam_trans_delta)                    
                    momentum_add = gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                            max_grad=0.005,
                                            min_opacity = 0.005, 
                                            extent = scene.cameras_extent, 
                                            args = opt,
                                            importance_score = importance_score,
                                            pruning_score = pruning_score)
                    
                    # Update max_n_gaussian
                    scheduler.update_momentum(momentum_add)
                    # Update render scale based on the DashGaussian resolution scheduler. 
                    render_scale = scheduler.get_res_scale(iteration)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if dataset.train_test_exp:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)
            
            # Update camera poses
            if iteration > opt.densify_from_iter and iteration % 300 == 0 and iteration < opt.densify_until_iter and opt.use_pose_optimization:
                update_global=True
                for view in scene.getTrainCameras():
                    _, global_transform = update_pose(view, cam_trans_delta, cam_rot_delta, global_transform, update_global)
                    update_global=False
                with open(os.path.join(scene.model_path, "global_transform.txt"), "w+") as f:
                    for row in global_transform.cpu().numpy():
                        f.write(" ".join([f"{x:.8f}" for x in row]) + "\n")
                cam_rot_delta.data.fill_(0)
                cam_trans_delta.data.fill_(0)
    
    with open(os.path.join(scene.model_path, "TRAIN_INFO"), "w+") as f:
        f.write("GS Number: {}\n".format(gaussians._scaling.shape[0]))


        
def eval(dataset, pipe, case_name, scene : Scene, renderFunc, renderArgs, iteration: int, time: int, log_file=None, global_transform=None):
    torch.cuda.empty_cache()
    config = {'name': 'test', 'cameras' : scene.getTestCameras()}
    (pipe, background, scale_factor, SPARSE_ADAM_AVAILABLE, overide_color, train_test_exp) = renderArgs
    if config['cameras'] and len(config['cameras']) > 0:
        l1_test = 0.0
        psnr_test = 0.0
        lpips_test = 0.0
        ssim_test = 0.0
        for idx, viewpoint in enumerate(config['cameras']):
            
            transform_viewpoint = update_pose_by_global(viewpoint, global_transform)
            image = torch.clamp(renderFunc(transform_viewpoint, scene.gaussians, *renderArgs, train_cameras = scene.getTrainCameras() if train_test_exp else None)["render"], 0.0, 1.0) 
            
            gt_image = torch.clamp(transform_viewpoint.original_image.to("cuda"), 0.0, 1.0)
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            lpips_test += lpips_fn(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            image_write = image.permute(1,2,0).detach().cpu().numpy()
            image_write = (image_write * 255).astype("uint8")
            os.makedirs(f"{scene.model_path}/test/", exist_ok = True)
            cv2.imwrite(os.path.join(f"{scene.model_path}/test/", "{}_{}_iter{:06d}.png".format(config['name'], transform_viewpoint.image_name, iteration)), cv2.cvtColor(image_write, cv2.COLOR_RGB2BGR))
        
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        lpips_test /= len(config['cameras'])
        ssim_test /= len(config['cameras'])
        print("\n[ITER {}] Evaluating {}sec: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, time, l1_test, psnr_test, lpips_test, ssim_test))
        
        if log_file is not None:
            data = {}
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            t_val = f"{float(time):.2f}"
            psnr_val = f"{float(psnr_test):.4f}"

            data[case_name] = {
                "PSNR": psnr_val,
                "time": t_val
            }

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)       


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--log_file", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.debug_from, args.log_file)

    # All done
    print("\nTraining complete.")
