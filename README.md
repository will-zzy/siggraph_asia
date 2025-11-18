## 0.install && quickly start

### 0.1.install

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

[AnySplat weights and config](https://huggingface.co/lhjiang/anysplat/tree/main) to `siggraph_asia/anySplat/ckpt`


[VGGT weights and config](https://huggingface.co/facebook/VGGT-1B/tree/main) to `siggraph_asia/vggt`

[Metric3D v2 weights](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_giant2_800k.pth) to `siggraph_asia/metric3D/weight`






### 0.2.quicikly start


Edit .vscode/full_train_and_eval.sh:

>Set `BASE_DIR` to the root directory containing all 13 scenes.<br>
>Set `EXP_DIR` to your experiment output directory.<br>
>Set `NAME` to the experiment name.

Then run training for all scenes:

```
chmod +x .vscode/full_train_and_eval.sh
.vscode/full_train_and_eval.sh
```

Per-scene quantitative results are aggregated in `${EXP_DIR}/metrics.json`, which records the rendering metrics and training time for all scenes.

Directory layout example:

```
  $EXP_DIR
    ├── 1747834320424
    ...
    ├── 1751090600427
    └── metrics.json
  ...
  $BASE_DIR
    └── 1751090600427
        ├── images_gt_downsampled
        ├── sparse/0
        └── train_test_split.json
```


## 1.Evaluation

Although evaluation runs during training, we also provide a unified evaluation script.

Use `.vscode/full_eval.sh` and edit:

`DATA_PATH` → root directory containing all 13 scenes

`EVALUATE_DIR` and `NAME` → the same values you used in the training script
```
chmod +x .vscode/full_eval.sh
.vscode/full_eval.sh
```
Please refer to `metrics.json` for the final time metrics, and `metrics_PSNR.json` for the rendering metrics.



