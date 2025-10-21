# 3DV-CASIA队伍-技术方案

## 0.数据集分析与方法概述

本次比赛数据集包含30个场景，包含以下特点：

> 1.SLAM结果较差：稀疏点云少，位姿不准；
>
> 2.虽然拍摄了30s~60s，但图像覆盖区域不大，且包含face forward和object centric拍摄
>
> 3.





经分析，我们对**Rasterization**中的`Forward->PreprocessCUDA`，`Barckward->RenderCUDA`进行了优化，并改进了三维表示使其更快收敛，同时应用了单目深度估计用于增添稠密点与监督训练，最后我们在训练过程中对位姿进行优化。

接下来我们详细介绍我们的上述模块





## 1.Methods

### 1.1Forward->Tighter BBox

在原版3DGS的方法中，我们借鉴了speedy-splat的解析前向法：

<img src="assets\w\image-20251021113719161.png" alt="image-20251021113719161" style="zoom:67%;" />

给定2D椭圆参数$Cov2D=\{a,b,c\}$，以及椭圆的质心$\mu$和椭球的不透明度$o$，椭圆参数化方程可描述为
$$
2\ln(255\cdot o)=t=ax_d^2+2bx_dy_d+cy_d^2
$$
并根据极值点方程$\partial{y_d}/\partial{x_d}=0$可求得$x$轴与$y$轴和椭圆的切点$x_{min}$/$x_{max}$，$y_{min}/y_{max}$，从而获得紧凑的`bounding box`（图1中的`SnugBox`）：
$$
y_{min/max}=\frac{-bx_d\pm\sqrt{(b^2-ac)x_d^2+tc}}{c},\;\;\;x_d=\pm\sqrt{\frac{b^2t}{(b^2-ac)a}}
$$




同时我们也在TNT数据集上对比了在`duplicateWithKeys`过程中使用`LOAD BALANCING`的



1.2.Backward->Per-Gaussian parrarelization

在Taming-GS中，



1.3.Representation->Neural Gaussians

应用了1.1和1.2中的加速方法之后，在4090上仍然没法在1min中内完成收敛，因此我们打算使用锚点来加速收敛，由于每个锚点维护一个feature，









1.4.Densification of Init points







1.5.Pose Optimization





2.测试结果



