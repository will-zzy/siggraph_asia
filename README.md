<div align="center">
  <h1>Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction</h1>
  
  <a href="https://gaplab.cuhk.edu.cn/projects/gsRaceSIGA2025/index.html"><img src="https://img.shields.io/badge/Official_Webpage-blue" alt="Official Website"></a>
  <a href="https://arxiv.org/abs/2601.19489"><img src="https://img.shields.io/badge/arXiv-2601.19489-b31b1b" alt="arXiv"></a>

  **Ziyu Zhang, Tianle Liu, Diantao Tu, Shuhan Shen $^\dagger$**

  Institute of Automation, Chinese Academy of Sciences; Wuhan University

  <h3>🏆 SIGA 2025 3DGS Challenge First Place 🏆</h3>
</div>

---

## Overview

We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge. Our solution includes several key points:

- Reverse per-Gaussian parallel optimization and compact forward splatting based on [Taming-GS](https://github.com/humansensinglab/taming-3dgs) and [Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat);
- Load-balanced tiling writing in function `duplicateWithKeys()`;
- Initialization from feed-forward 3DGS model based on [AnySplat](https://github.com/InternRobotics/AnySplat);
- Multi-view consistency-guided densification and pruning strategy inspired by [Fast-GS](https://github.com/fastgs/FastGS);
- Supervise the rendered depth by depth estimator [Metric3D](https://github.com/YvanYin/Metric3D);
- Optional global transform on camera extrinsics.

In addation, We tried using [Neural Gaussians](https://github.com/city-super/Scaffold-GS) to replace original 3DGS for fast convergence with few parameters. Switch to the `use_scaffold_release` branch for details.

## TODO

- [x] Release the 3dgs code
- [ ] Add Load-balanced tiling writing
- [ ] Release the neural gaussian code

## Getting Started

### 1. Prerequisites

- Ubuntu 22.04
- one 4090 GPU
- multi-core CPU (recommend > 20)

### 2. Installation

```bash
# 1) Create and activate a Conda environment
conda create -n 3dv_gs python=3.10
conda activate 3dv_gs

# 2) Install xFormers built for your CUDA
# Replace the URL with the one matching your local CUDA version. We use CUDA 12.4 and torch 2.5.1
# xformers==0.0.29 will automatically pull torch==2.5.1 as a dependency.
pip install -U xformers==0.0.29 --index-url https://download.pytorch.org/whl/cu124

# 3) Install torchvision that matches your torch
# torch 2.5.1 pairs with torchvision 0.20.1. We recommend using a prebuilt wheel:
# https://download.pytorch.org/whl/torchvision/
wget <the-correct-torchvision-wheel-from-the-link-above>
pip install torchvision-xxxx.whl

# 4) Verify versions and pin them
# Run `pip list` to confirm the installed torch/torchvision versions,
# then put those exact versions into constraints.txt.

# 5) Install project requirements with pinned versions (no build isolation)
pip install -r requirements.txt -c constraints.txt --no-build-isolation

# 6) Install local extensions (no build isolation)
pip install submodules/diff-gaussian-rasterization \
            submodules/simple-knn \
            submodules/fused-ssim \
            submodules/lanczos-resampling \
            --no-build-isolation
```

Also download the following weights/configs:

- [AnySplat weights and config](https://huggingface.co/lhjiang/anysplat/tree/main) to `./anySplat/ckpt`
- [VGGT weights and config](https://huggingface.co/facebook/VGGT-1B/tree/main) to `./vggt`
- [Metric3D v2 weights](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_giant2_800k.pth) to `./metric3D/weight`



## Quick Start

### 1. Training

#### Modify Dataset Path

Edit `.vscode/full_train_and_eval.sh`:

- Set `BASE_DIR` to the **root directory** containing all scenes.
- Set `EXP_DIR` to your experiment **output directory**.

Then run training script for all scenes:

```bash
chmod +x .vscode/full_train_and_eval.sh
.vscode/full_train_and_eval.sh
```

Per-scene quantitative results are aggregated in `${EXP_DIR}/metrics_train.json`, which records the rendering metrics and training time for all scenes.

#### Directory Layout Example

```
  $EXP_DIR
    ├── <scene_1>
    ...
    ├── <scene_n>
    └── metrics_train.json
  ...
  $BASE_DIR
    └── <scene_1>
        ├── images
        ├── sparse/0
        └── train_test_split.json
```

#### Heads-up on hardware variability

- **Rare VRAM out-of-bounds on RTX 4090**: We observed a very small probability of a memory out-of-bounds issue on 4090 GPUs, which we believe is caused by excessive VRAM pressure. **If this occurs, please re-run the affected scene—in our tests, the issue is non-deterministic and typically disappears on retry**.
- **CPU Core Amount and Ruselts**: The training loop will stop after 1 minute or when the maximum number of iterations is reached. We observe the training time to reach the same iteration may be different on cpus with different core numbers. **Please modify the `$iterations` according to your device, so that the overall training time is about one minute.**

### 2. Evaluation

Edit `.vscode/full_eval.sh`:

- Set `DATA_PATH` to the **root directory** containing all scenes.
- Set `EVALUATE_DIR` to same experiment **output directory** as above.

Then run eval script for all scenes:

```bash
chmod +x .vscode/full_eval.sh
.vscode/full_eval.sh
```
Please refer to `metrics.json` for the final time metrics and PSNR metrics.


## Acknowledgments

We thank the above excellent open source repositories for sharing and the support of the competition organizers.

## Citation

```
@article{zhang2026fastconverging3dgaussian,
  title={Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction}, 
  author={Ziyu Zhang and Tianle Liu and Diantao Tu and Shuhan Shen},
  journal={arXiv preprint arXiv:2601.19489},
  year={2026},
}
```