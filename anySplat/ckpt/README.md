---
license: mit
pipeline_tag: image-to-3d
---

# AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views

[![Project Website](https://img.shields.io/badge/AnySplat-Website-4CAF50?logo=googlechrome&logoColor=white)](https://city-super.github.io/anysplat/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/pdf/2505.23716)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Code-FFD700?logo=github)](https://github.com/OpenRobotLab/AnySplat)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/lhjiang/anysplat)


## Quick Start

See the Github repository: https://github.com/OpenRobotLab/AnySplat regarding installation instructions.

The model can then be used as follows:

```python
from pathlib import Path
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image

# Load the model from Hugging Face
model = AnySplat.from_pretrained("anysplat_ckpt_v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Load and preprocess example images (replace with your own image paths)
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"] 
images = [process_image(image_name) for image_name in image_names]
images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
b, v, _, h, w = images.shape

# Run Inference
gaussians, pred_context_pose = model.inference((images+1)*0.5)

pred_all_extrinsic = pred_context_pose['extrinsic']
pred_all_intrinsic = pred_context_pose['intrinsic']
save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, image_folder, model.decoder)

```

## Citation

```
@article{jiang2025anysplat,
  title={AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views},
  author={Jiang, Lihan and Mao, Yucheng and Xu, Linning and Lu, Tao and Ren, Kerui and Jin, Yichen and Xu, Xudong and Yu, Mulin and Pang, Jiangmiao and Zhao, Feng and others},
  journal={arXiv preprint arXiv:2505.23716},
  year={2025}
}
```

## License

The code and models are licensed under the [MIT License](LICENSE).