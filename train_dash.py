import os
import numpy as np
import torch
import torchvision
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.camera_utils import update_pose
# from gaussian_renderer import render, network_gui
from gaussian_renderer import prefilter_voxel, render, network_gui, render_origin, render_simp
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import GaussianModel_origin
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.schedule_utils import TrainingScheduler
from utils.robotic_utils import sh_rotation, matrix_to_quaternion, quaternion_to_matrix, load_ply, save_ply


import cv2
import csv
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

UPLOAD_IMG = False
from typing import NamedTuple
class CamNamePoseInfo(NamedTuple):
    image_name: str
    R: np.array
    T: np.array
    pose: np.array # 注意anySplat出来的c2w
def anySplat(dataset, opt, pipe):
    
    from pathlib import Path
    import torch
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for loading local AnySplat checkpoints. "
            "Please install it with `pip install safetensors`."
        ) from exc
    from anySplat.model.model.anysplat import AnySplat
    from utils.image_utils import process_image

    # Load the model from Hugging Face
    # model = AnySplat.from_pretrained("./anySplat/ckpt/model.safetensors")
    # model = AnySplat.from_pretrained("lhjiang/anysplat")
    ckpt_root = Path("./anySplat/ckpt")
    config_path = ckpt_root / "config.json"
    weights_path = ckpt_root / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"AnySplat config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"AnySplat weights not found: {weights_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    encoder_cfg_dict = config_dict.get("encoder_cfg")
    decoder_cfg_dict = config_dict.get("decoder_cfg")
    if encoder_cfg_dict is None or decoder_cfg_dict is None:
        raise ValueError(f"Invalid AnySplat config structure: {config_path}")

    from anySplat.model.encoder.anysplat import EncoderAnySplatCfg
    from anySplat.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg

    vggt_override = os.environ.get("ANY_SPLAT_VGGT_WEIGHTS")
    if vggt_override:
        encoder_cfg_dict["pretrained_weights"] = vggt_override

    encoder_cfg = EncoderAnySplatCfg(**encoder_cfg_dict)
    decoder_cfg = DecoderSplattingCUDACfg(**decoder_cfg_dict)
    model = AnySplat(encoder_cfg, decoder_cfg)

    state_dict = load_file(str(weights_path))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[AnySplat] Missing keys when loading weights: {missing_keys}")
    if unexpected_keys:
        print(f"[AnySplat] Unexpected keys when loading weights: {unexpected_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load and preprocess example images (replace with your own image paths)
    
    images_dir = os.path.join(dataset.source_path, "images")
    from scene.colmap_loader import read_extrinsics_text, qvec2rotmat
    cameras_extrinsic_file = os.path.join(dataset.source_path, "sparse/0", "images.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    imgs = []
    cams_unsorted = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec).reshape(-1, 1)
        c2w = np.concatenate([R, T], axis=1) # 注意anySplat出来的是c2w
        c2w = np.linalg.inv(np.concatenate([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0))
        cams_unsorted.append(CamNamePoseInfo(image_name=extr.name, R=R, T=T, pose=c2w))
    cams = sorted(cams_unsorted.copy(), key = lambda x : x.image_name)
    
    # image_names = os.listdir(images_dir)
    slam_c2ws = []
    for cam in cams:
        img_name = cam.image_name
        imgs.append(os.path.join(images_dir, img_name))
        slam_c2ws.append(cam.pose[None])
    slam_c2ws = np.concatenate(slam_c2ws, axis=0)
    images = [process_image(image_name) for image_name in imgs]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape

    # Run Inference
    gs, pred_context_pose = model.inference((images+1)*0.5)
    # gaussians.mean
    # downsample = 8
    downsample = pipe.FF_downsample # for scaffold's anchor
    gaussians = GaussianModel_origin(4, "sparse_adam")
    gaussians._xyz = gs.means[0, ::downsample, ...]
    gaussians._scaling = gaussians.scaling_inverse_activation(gs.scales[0, ::downsample, ...])
    gaussians._opacity = gaussians.inverse_opacity_activation(gs.opacities[0, ::downsample, ...][..., None])
    gaussians._rotation = gs.rotations[0, ::downsample, ...]
    gaussians._features_dc = gs.harmonics[0, ::downsample, :, :1].transpose(1, 2)
    gaussians._features_rest = gs.harmonics[0, ::downsample, :, 1:].transpose(1, 2)
    
    del gs, model
    torch.cuda.empty_cache()
    
    # gaussians.save_ply("/home/zzy/lib/siggraph_asia/tmp.ply")
    
    pred_all_extrinsic = pred_context_pose['extrinsic'][0] # [N_nums, 4, 4]
    pred_all_intrinsic = pred_context_pose['intrinsic'][0] # [N_nums, 3, 3]
    
    transform, stats, gaussians_aligned = align(gaussians=gaussians, anysplat_traj=pred_all_extrinsic, slam_traj=slam_c2ws)
    return gaussians_aligned, pred_all_extrinsic, pred_all_intrinsic
    # save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)


def align(gaussians: GaussianModel_origin, anysplat_traj, slam_traj, ply_path=None):
    """
    Align AnySplat trajectory to SLAM trajectory, apply the estimated similarity
    transform to the Gaussian model, and export the updated point cloud.
    """
    from tnt_eval.registration import (
        estimate_similarity_transform,
        apply_similarity_transform_to_gaussians,
        _trajectory_to_pose_tensor,
    )

    if gaussians is None:
        raise ValueError("Gaussian model is required for alignment.")

    transform, stats = estimate_similarity_transform(anysplat_traj, slam_traj)
    
    
    # align_out = apply_similarity_transform_to_gaussians(
    #     gaussians, # {"scale": scale, "rotation_quaternion": rotation_matrix_to_quaternion(rotation)}
    #     transform,
    #     rotation=stats.get("rotation"),
    #     translation=stats.get("translation"),
    #     scale=stats.get("scale"),
    # )
    rotation=torch.tensor(stats.get("rotation"),device=gaussians._xyz.device).float()
    translation=torch.tensor(stats.get("translation"),device=gaussians._xyz.device).float()
    scale=torch.tensor(stats.get("scale"),device=gaussians._xyz.device).float()
    
    
    # rotation = transform_matrix[:3, :3]
    # translation = transform_matrix[None, :3, 3]
    # scale_check = (rotation @ rotation.T)[0, 0] ** 0.5
    # rotation_pure = rotation / scale_check
    xyz = gaussians._xyz
    xyz = scale * (xyz @ rotation.T) + translation
    gaussians._xyz.data.copy_(xyz)
    del xyz
    
    scaling = gaussians._scaling
    scaling = scaling + torch.log(scale * 4)
    gaussians._scaling.data.copy_(scaling)
    del scaling
    
    rots = gaussians._rotation
    rot_matrices = quaternion_to_matrix(rots)
    new_rot_matrices = rotation @ rot_matrices
    rots = matrix_to_quaternion(new_rot_matrices)
    rots = rots / torch.norm(rots, dim=-1, keepdim=True)
    gaussians._rotation.data.copy_(rots)
    del rots, rot_matrices, new_rot_matrices
    
    features_extra = gaussians._features_rest
    features_extra = sh_rotation(features_extra.reshape((features_extra.shape[0],3,-1)), gaussians._features_dc, rotation)
    gaussians._features_rest.data.copy_(features_extra.reshape((features_extra.shape[0],-1,3)))
    del features_extra

    # gaussian_new = GaussianModel_origin(4, "sparse_adam")
    # gaussian_new._xyz = xyz
    # gaussian_new._scaling = scaling
    # gaussian_new._rotation = rots
    # gaussian_new._opacity = gaussians._opacity
    # gaussian_new._features_dc = features_dc
    # gaussian_new._features_rest = features_extra
    
    
    # xyz_np = xyz.cpu().numpy()
    # features_dc_np = features_dc.reshape(-1, 3).cpu().numpy()
    # features_extra_np = features_extra.reshape(-1, 45).cpu().numpy()
    # opacities_np = opacities.cpu().numpy()
    # scales_np = scales.cpu().numpy()
    # rots_np = rots.cpu().numpy()

    if ply_path is not None:
        gaussians.save_ply(ply_path)

    # Prepare debug outputs (plot + point clouds) to inspect alignment quality.
    try:
        source_tensor = _trajectory_to_pose_tensor(anysplat_traj)
        target_tensor = _trajectory_to_pose_tensor(slam_traj)
    except Exception as exc:
        print(f"[align] Unable to convert trajectories for visualization: {exc}")
        return transform, stats

    rotation = stats.get("rotation")
    translation = stats.get("translation")
    scale = stats.get("scale")

    if rotation is None or translation is None or scale is None:
        print("[align] Missing rotation/translation/scale in stats; skip visualization.")
        return transform, stats

    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    scale = float(scale)

    source_centers = source_tensor[:, :3, 3].cpu().numpy()
    aligned_centers = (rotation @ source_centers.T).T
    aligned_centers = scale * aligned_centers + translation
    gt_centers = target_tensor[:, :3, 3].cpu().numpy()

    # base_dir = Path(ply_path).parent if ply_path is not None else Path.cwd() / "align_outputs"
    # base_dir.mkdir(parents=True, exist_ok=True)
    # base_name = Path(ply_path).stem if ply_path is not None else "align"
    # unique_tag = uuid.uuid4().hex[:8]
    # prefix = f"{base_name}_{unique_tag}"

    # png_path = base_dir / f"{prefix}_traj.png"
    # aligned_ply_path = base_dir / f"{prefix}_aligned_traj.ply"
    # gt_ply_path = base_dir / f"{prefix}_gt_traj.ply"

    # Save 3D trajectory plot.
    # try:
    #     import matplotlib
    #     import sys as _sys

    #     if "matplotlib.pyplot" not in _sys.modules:
    #         matplotlib.use("Agg")  # Ensure headless plotting.
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    #     fig = plt.figure(figsize=(6, 6))
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.plot(
    #         aligned_centers[:, 0],
    #         aligned_centers[:, 1],
    #         aligned_centers[:, 2],
    #         label="aligned traj",
    #         color="tab:blue",
    #     )
    #     ax.plot(
    #         gt_centers[:, 0],
    #         gt_centers[:, 1],
    #         gt_centers[:, 2],
    #         label="gt traj",
    #         color="tab:orange",
    #     )
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("Trajectory Alignment")
    #     ax.legend()
    #     fig.tight_layout()
    #     fig.savefig(png_path, dpi=300)
    #     plt.close(fig)
    #     stats["trajectory_plot_png"] = str(png_path)
    # except Exception as exc:
    #     print(f"[align] Failed to save trajectory plot: {exc}")

    # # Save aligned and GT trajectories as point clouds.
    # try:
    #     import open3d as o3d

    #     aligned_pcd = o3d.geometry.PointCloud()
    #     aligned_pcd.points = o3d.utility.Vector3dVector(
    #         np.asarray(aligned_centers, dtype=np.float64)
    #     )
    #     o3d.io.write_point_cloud(str(aligned_ply_path), aligned_pcd)

    #     gt_pcd = o3d.geometry.PointCloud()
    #     gt_pcd.points = o3d.utility.Vector3dVector(
    #         np.asarray(gt_centers, dtype=np.float64)
    #     )
    #     o3d.io.write_point_cloud(str(gt_ply_path), gt_pcd)

    #     stats["aligned_traj_ply"] = str(aligned_ply_path)
    #     stats["gt_traj_ply"] = str(gt_ply_path)
    # except Exception as exc:
    #     print(f"[align] Failed to save trajectory point clouds: {exc}")

    return transform, stats, gaussians


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, log_file=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians = None
    if args.useFF:
        gaussians, pred_all_extrinsic, pred_all_intrinsic = anySplat(lp.extract(args), op.extract(args), pp.extract(args))
        # align()
        gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device=gaussians.get_xyz.device)
        gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device=gaussians.get_xyz.device)
        scene = Scene(dataset, gaussians, pipe, shuffle=False)
        
    
    else:
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        anchor_xyz, pred_all_extrinsic, pred_all_intrinsic = anySplat(lp.extract(args), op.extract(args), pp.extract(args))
        scene = Scene(dataset, gaussians, pipe, anchor_xyz, shuffle=False)
    
    
    
    
    gaussians.training_setup(opt)
    if checkpoint:
        
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    # optim_start = torch.cuda.Event(enable_timing = True)
    # optim_end = torch.cuda.Event(enable_timing = True)
    # total_time = 0.0

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Init DashGaussian scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians, 
                                  [cam.original_image for cam in scene.getTrainCameras()])
    render_scale = scheduler.get_res_scale(1)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 添加时间计数器，在iteration开始后开始计时
    start_time = time.time()
    eval_time = None
    time_save_iterations = [30, 60]  # 30秒和60秒时保存
    
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        # vind = viewpoint_indices.pop(rand_idx)
        # pred_w2c = torch.inverse(pred_all_extrinsic[rand_idx, ...])
        # viewpoint_cam.R = pred_w2c[:3, :3].T
        # viewpoint_cam.T = pred_w2c[:3, 3]
        # viewpoint_cam.update_W2C = False
        # viewpoint_cam.update_W2I = False
        # viewpoint_cam.update_center = False

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if not pipe.useFF:
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        # Rescale GT image for DashGaussian
        gt_image = viewpoint_cam.original_image.cuda()
        if render_scale > 1:
            gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", 
                                                       recompute_scale_factor=True, antialias=True)[0]
        if not pipe.useFF:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_image.shape[-2:], visible_mask=voxel_visible_mask, retain_grad=retain_grad)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        else:
            render_pkg = render_origin(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_image.shape[-2:])
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # if viewpoint_cam.alpha_mask is not None:
        #     alpha_mask = viewpoint_cam.alpha_mask.cuda()
        #     if render_scale > 1:
        #         alpha_mask = torch.nn.functional.interpolate(alpha_mask[None], scale_factor=1/render_scale, 
        #                                                 mode="bilinear", recompute_scale_factor=True, antialias=True)[0]
        #     image *= alpha_mask

        # Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0 # TODO: maybe add depth supervision
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable: # 加速收敛
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if iteration % 200 == 0:
                image_write = image.permute(1,2,0).detach().cpu().numpy()
                image_write = (image_write * 255).astype("uint8")
                os.makedirs(f"{scene.model_path}/test/", exist_ok = True)
                cv2.imwrite(os.path.join(f"{scene.model_path}/test/", "iter{:06d}_{}.png".format(iteration, viewpoint_cam.image_name)), cv2.cvtColor(image_write, cv2.COLOR_RGB2BGR))
                
        
        
        
        
        
        
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                if pipe.useFF:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "N_GS": f"{gaussians.get_scaling.shape[0]}", "N_MAX": f"{scheduler.max_n_gaussian}", "R": f"{render_scale}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "N_GS": f"{gaussians.get_scaling.shape[0] * dataset.n_offsets}", "N_MAX": f"{scheduler.max_n_gaussian}", "R": f"{render_scale}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            ########## TODO 是否要删除test视角渲染？
            iter_time = iter_start.elapsed_time(iter_end)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            ##########
                
            #  检查时间计数器，30秒和60秒时保存结果
            elapsed_time = time.time() - start_time
            if elapsed_time >= 30 and 30 in time_save_iterations:
                eval_start_time = time.time()
                print(f"\n[ITER {iteration}] Evaluating Gaussians at 30 seconds")
                eval(scene, render, render_origin, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), iteration, 30, log_file)
                eval_time = time.time() - eval_start_time
                time_save_iterations.remove(30)  # 移除已保存的时间点，避免重复保存
            elif eval_time != None and elapsed_time - eval_time >= 60 and 60 in time_save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians at 60 seconds")
                eval(scene, render, render_origin, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), iteration, 60, log_file)
                scene.save(iteration)
                time_save_iterations.remove(60)  # 移除已保存的时间点，避免重复保存
                # 到60秒后退出训练
                sys.exit(0)

            # optim_start.record()



            if not pipe.useFF: # 不使用FF，使用scaffold-GS
                # Densification
                if iteration < opt.densify_until_iter and iteration > opt.start_stat:
                    # Keep track of max radii in image-space for pruning
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, scheduler=scheduler)
                    # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    #     # Apply DashGaussian primitive scheduler to control densification.
                    #     densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_anchor.shape[0], render_scale)
                    #     momentum_add = gaussians.prune_and_densify(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 
                    #                                                size_threshold, radii, densify_rate=densify_rate)
                    #     # Update max_n_gaussian
                    #     scheduler.update_momentum(momentum_add)
                    #     # Update render scale based on the DashGaussian resolution scheduler. 
                    #     render_scale = scheduler.get_res_scale(iteration)

                    # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    #     gaussians.reset_opacity()
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()

                # Optimizer step
                if iteration < opt.iterations:
                    # gaussians.exposure_optimizer.step()
                    # gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    # if use_sparse_adam:
                    #     visible = radii > 0
                    #     gaussians.optimizer.step(visible, radii.shape[0])
                    #     gaussians.optimizer.zero_grad(set_to_none = True)
                    # else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    viewpoint_cam.pose_optimizer.step()
                    viewpoint_cam.pose_optimizer.zero_grad(set_to_none = True)

                # optim_end.record()
                # torch.cuda.synchronize()
                # optim_time = optim_start.elapsed_time(optim_end)
                # total_time += (iter_time + optim_time) / 1e3

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            
            
            
            
            elif pipe.useFF: # 使用原版的
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # Apply DashGaussian primitive scheduler to control densification.
                        densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_xyz.shape[0], render_scale)
                        momentum_add = gaussians.prune_and_densify(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 
                                                                   size_threshold, radii, densify_rate=densify_rate)
                        # Update max_n_gaussian
                        scheduler.update_momentum(momentum_add)
                        # Update render scale based on the DashGaussian resolution scheduler. 
                        render_scale = scheduler.get_res_scale(iteration)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    # gaussians.exposure_optimizer.step()
                    # gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

                    viewpoint_cam.pose_optimizer.step()
                    viewpoint_cam.pose_optimizer.zero_grad(set_to_none = True)
            
            
            if iteration > 1000 and iteration % 400 == 0 and iteration < opt.update_until :
                # update_pose(viewpoint_cam)
                for view in scene.getTrainCameras():
                    update_pose(view)
    with open(os.path.join(scene.model_path, "TRAIN_INFO"), "w+") as f:
        # f.write("Training Time: {:.2f} seconds, {:.2f} minutes\n".format(total_time, total_time / 60.))
        f.write("GS Number: {}\n".format(gaussians.get_scaling.shape[0]))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
    
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        (pipe, background, scale_factor, SPARSE_ADAM_AVAILABLE, overide_color, train_test_exp) = renderArgs
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, pipe, background, 1.0, None)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if UPLOAD_IMG and tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    # image_write = image.permute(1,2,0).detach().cpu().numpy()
                    # image_write = (image_write * 255).astype("uint8")
                    # os.makedirs(f"{scene.model_path}/test/", exist_ok = True)
                    # cv2.imwrite(os.path.join(f"{scene.model_path}/test/", "iter{:06d}_{}_{}.png".format(iteration, config['name'], viewpoint.image_name)), cv2.cvtColor(image_write, cv2.COLOR_RGB2BGR))
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_scaling.shape[0], iteration)
        torch.cuda.empty_cache()
        
def eval(scene : Scene, renderFunc, renderFunc2, renderArgs, iteration: int, time: int, log_file=None):
    torch.cuda.empty_cache()
    config = {'name': 'test', 'cameras' : scene.getTestCameras()}
    (pipe, background, scale_factor, SPARSE_ADAM_AVAILABLE, overide_color, train_test_exp) = renderArgs
    if config['cameras'] and len(config['cameras']) > 0:
        l1_test = 0.0
        psnr_test = 0.0
        lpips_test = 0.0
        ssim_test = 0.0
        for idx, viewpoint in enumerate(config['cameras']):
            try:
                voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, pipe, background, 1.0, None)
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
            except:
                image = torch.clamp(renderFunc2(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
            # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            if train_test_exp:
                image = image[..., image.shape[-1] // 2:]
                gt_image = gt_image[..., gt_image.shape[-1] // 2:]
            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            lpips_test += lpips_fn(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            image_write = image.permute(1,2,0).detach().cpu().numpy()
            image_write = (image_write * 255).astype("uint8")
            # os.makedirs(f"{scene.model_path}/test/", exist_ok = True)
            # cv2.imwrite(os.path.join(f"{scene.model_path}/test/", "{}_{}_iter{:06d}.png".format(config['name'], viewpoint.image_name, iteration)), cv2.cvtColor(image_write, cv2.COLOR_RGB2BGR))
        
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        lpips_test /= len(config['cameras'])
        ssim_test /= len(config['cameras'])
        print("\n[ITER {}] Evaluating {}sec: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, time, l1_test, psnr_test, lpips_test, ssim_test))
        
        # 记录到文件
        if log_file is not None:
            with open(log_file, 'a', newline='') as csvfile:
                log_writer = csv.writer(csvfile)
                if time == 30:
                    log_writer.writerow([scene.model_path.split('/')[-3], 30, f"{ssim_test:.4f}", f"{lpips_test:.4f}", f"{psnr_test:.4f}"])
                elif time == 60:
                    log_writer.writerow([scene.model_path.split('/')[-3], 60, f"{ssim_test:.4f}", f"{lpips_test:.4f}", f"{psnr_test:.4f}"])
            
    torch.cuda.empty_cache()
    

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
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * (i + 1) for i in range(30)])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--useFF", type=bool, default=False)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--log_file", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.log_file)

    # All done
    print("\nTraining complete.")
