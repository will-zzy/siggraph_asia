## 0.install && quickly start

**由于赛题有不明确之处，因此我们准备了两套代码以应对不同的测评方法，两套代码均使用同一套环境，具体区别在1.5位姿优化中介绍**
### 0.1.install

```bash
conda create -n 3dv_gs python=3.10
conda activate 3dv_gs
pip install torch=xxx torchvision=xxx # 请根据自己的cuda版本选择

pip install -r requirements.txt
```

同时需要下载[AnySplat的权重与配置文件](https://huggingface.co/lhjiang/anysplat/tree/main)（下载文件名为`model.safetensors`和`config.json`）到`siggraph_asia/anySplat/ckpt`下

以及[VGGT的权重与配置文件](https://huggingface.co/facebook/VGGT-1B/tree/main)（`model.safetensors`和`config.json`）到任意目录下，并在`.vscode/run_all.sh`中，将`VGGT_PATH`的路径改为该下载目录



### 0.2.quicikly start

将`run_all.sh`中的`ROOT_BASE`改为场景文件的数据集的父目录，`NAME`为实验名称，`VGGT_PATH`为上述VGGT权重和配置文件下载目录。并通过下面的指令运行所以实验的训练步骤：

```
chmod +x .vscode/run_all.sh
./.vscode/run_all.sh
```

每个场景的实验定量结果在`eval_logs/anysplat_scaffold/`获得，分别记录了每个场景的定量评价指标和所有场景的平均定量指标。






## 1.Methods

### 1.1Forward->Tighter BBox

在原版3DGS的方法中，我们借鉴了speedy-splat的解析前向法：

<img src="assets/w/image-20251021113719161.png" alt="image-20251021113719161" style="zoom:67%;" />

​                  图一：Tighter BBox示意图



给定2D椭圆参数$Cov2D=\{a,b,c\}$，以及椭圆的质心$\mu$和椭球的不透明度$o$，椭圆参数化方程可描述为
<br>

$2\ln(255\cdot o)=t=ax_d^2+2bx_dy_d+cy_d^2$

<br>
并根据极值点方程$\partial{y_d}/\partial{x_d}=0$可求得$x$轴与$y$轴和椭圆的切点$x_{min}$/$x_{max}$，$y_{min}/y_{max}$，从而获得紧凑的`bounding box`（图1中的`SnugBox`）：

<br>

$y_{min/max}=\frac{-bx_d\pm\sqrt{(b^2-ac)x_d^2+tc}}{c},\;\;\;x_d=\pm\sqrt{\frac{b^2t}{(b^2-ac)a}}$

<br>

随后，在`SnugBox`所占据的`rectangle tiles`中，我们遍历列`tiles`，并根据列`tile`的左右边界坐标（e.g. $x=x_{tmin},\;\;x=x_{tmax}$）闭式计算出与椭圆的交点，从而确定该列有哪些`tile`与椭圆相交，并顺序写入`tile id`和`depth value`。





### 1.2.Backward->Per-Gaussian parrarelization

Taming-GS提出使用per-Gaussian而不是per-pixel的反向传播策略。具体来说，传统的3DGS在反向传播时每个线程负责一个像素，并顺序遍历叠在该像素上的所有高斯，并对梯度进行原子加法，这在不同线程中会引起严重的竞态。在一般场景中，一个tile上往往叠加了几百个splats，这也会使得并行数（256）小于顺序遍历数。而Taming-GS提出每个线程负责一个splat，计算完当前tile所有像素对于该splat的梯度贡献后再进行原子加法，大大减小了线程竞态；

<img src="assets/w/image-20251021164509192.png" alt="image-20251021164509192" style="zoom:25%;" />    <img src="assets/w/image-20251021164524524.png" alt="image-20251021164524524" style="zoom:25%;" /> 

图二：per-gaussian 反向传播示意图



我们在前向渲染时，**每个像素**每渲染32个splats记录一次反向传播时所需的透射率`T`和所blending的颜色`C`，从而在反向传播时，每个warp可以独立地对自己组内的splat递归更新梯度，如图3（右）所示，不同颜色表示不同的warp，每个warp根据前向时所记录的`T`和`C`在warp内遍历当前tile所有像素，递归更新每个splat的梯度

<img src="assets/w/image-20251021165143361.png" alt="image-20251021165143361" style="zoom:25%;" />                <img src="assets/w/image-20251021165116057.png" alt="image-20251021165116057" style="zoom:25%;" />

图三：per-gaussian 反向传播示意图



将`Forward`与`Backward`结合，我们在TNT数据集上测试了训练时长（30000轮迭代），同时我们也比较了在`duplicateWithKeys`过程中使用`LOAD BALANCING`的时长：

| **TNT /s (4090)**                    | **Barn** | **Truck** | **Ignatius** | **Meeting** | **Caterp** |
| :----------------------------------- | :------: | :-------: | :----------: | :---------: | :--------: |
| 3DGS (baseline)                      |   638    |    611    |     618      |     574     |    615     |
| w/ Backward                          |   191    |    173    |     181      |     145     |    183     |
| w/ Backward & Forward                |   180    |  **159**  |   **173**    |   **137**   |  **171**   |
| w/ Backward & Forward & Load Balance | **176**  |    163    |     177      |     141     |    171     |

### 1.3.Representation->Neural Gaussians

应用了1.1和1.2中的加速方法之后，我们发现在4090上仍然没法在1min中内完成收敛，哪怕提前截止（20000轮），时间也在1min开外，因此我们打算更改表达方式进行加速。

我们分析，由于在原本的表达中，每个splat为一个单独的叶子节点，这使得每个splat并不能共享优化信息，使得优化效率低下。为此我们引入了`Scaffold-GS`作为表达，使用`anchor`的特征来inference `splats`的属性，这大大减少了优化参数数量，并使得收敛更快。

<img src="assets/w/image-20251021171748579.png" alt="image-20251021171748579" style="zoom:40%;" />

图四：scaffold-GS（左）和vanilla-GS（右）对比图

**注意，由于我们使用了scaffold-GS表达，因此point_cloud.ply是scaffold-GS的点云(神经点云)，同时神经点云的rgb是随视角相关的，不能直接转换为高斯点云，因此请使用我们的渲染器渲染，我们提供了一个简易的渲染样例`transformScaffold2Gaussian.py`，该文件由`.vscode/eval_global.sh`运行。**

### 1.4.Feedforward Initial Points

我们使用了AnySplat的输出点位置，降采样后作为初始的`anchor`位置，

<img src="assets/w/image-20251024170415411.png" alt="image-20251024170415411" style="zoom:40%;" />

图五：稠密化初始点（左）与不稠密化（右）对比图

**需要注意一点，anySplat和VGGT并不是必须的，使用轻量的深度估计模型增加初始点数量也是可行的**

### 1.5.Pose Optimization

由于数据集的位姿十分不准确，我们进行了位姿优化



<img src="assets/w/image-20251021171815382.png" alt="image-20251021171815382" style="zoom:40%;" />

图六：使用位姿优化（右）与不使用位姿优化（左）对比图

**由于我们进行了位姿优化，而测试视角还是未经优化过的位姿，如何将测试视角对齐到优化后坐标系下是一个难点，因此我们准备了两套方案。**<br>

**第一套，每个相机独自维护一个可学习的偏移量，这套方案PSNR较高，但无法评测官方给定视角下的图片（尽管渲染质量更高）。**<br>

**第二套，仅维护一个global的偏移量（4x4矩阵），当使用和官方数据集初始位姿同一坐标系下的新视角渲染时，将这个4x4矩阵左乘新视角的W2C，具体可见代码我们第二份代码siggraph_asia中`utils/camera_utils.py`的`update_pose_by_global`函数，global transform可在`model_path`下的`global_transform.txt`中读取。**


**由于没有测试集的准确位姿，我们仅评估训练集的PSNR**

## 2.测试结果

| case           | time   | SSIM   | LPIPS   | PSNR    |
|---------------:|:------:|:------:|:-------:|:-------:|
| 1747834320424  | 43.34s | 0.9137 | 0.1793  | 28.7152 |
| 1748153841908  | 55.24s | 0.9519 | 0.1450  | 31.9326 |
| 1748165890960  | 60.00s | 0.8176 | 0.2902  | 24.0374 |
| 1748242779841  | 58.48s | 0.8995 | 0.2086  | 28.3987 |
| 1748243104741  | 60.00s | 0.9433 | 0.1318  | 30.8882 |
| 1748265156579  | 57.79s | 0.8766 | 0.1907  | 28.1803 |
| 1748411766587  | 60.00s | 0.7958 | 0.2257  | 24.2349 |
| 1748422612463  | 44.46s | 0.9764 | 0.1387  | 34.7215 |
| 1748686118256  | 60.00s | 0.7472 | 0.3474  | 23.9466 |
| 1748689420856  | 60.00s | 0.6749 | 0.4103  | 22.6885 |
| 1748748200211  | 50.75s | 0.8185 | 0.2607  | 25.6982 |
| 1748781144981  | 42.67s | 0.8938 | 0.2213  | 26.3845 |
| 1748833935627  | 60.00s | 0.8630 | 0.2378  | 26.0193 |
| 1749369580718  | 40.92s | 0.9398 | 0.2110  | 29.4450 |
| 1749449291156  | 43.75s | 0.9198 | 0.1611  | 29.9286 |
| 1749606908096  | 41.90s | 0.9330 | 0.1712  | 28.8225 |
| 1749803955124  | 40.95s | 0.9331 | 0.1978  | 30.7032 |
| 1749864076665  | 46.46s | 0.9312 | 0.2349  | 32.0199 |
| 1749972078570  | 60.00s | 0.7297 | 0.3543  | 23.2883 |
| 1749974151642  | 59.32s | 0.9182 | 0.1493  | 27.8556 |
| 1749975239137  | 42.72s | 0.9204 | 0.2122  | 27.8274 |
| 1749977115648  | 43.84s | 0.9170 | 0.1774  | 28.4371 |
| 1750342099701  | 60.00s | 0.8224 | 0.2417  | 25.2593 |
| 1750342304509  | 60.00s | 0.8983 | 0.2665  | 28.7164 |
| 1750343446362  | 60.00s | 0.8187 | 0.2919  | 24.8870 |
| 1750383597053  | 47.32s | 0.9558 | 0.1435  | 33.4527 |
| 1750578027423  | 42.33s | 0.9741 | 0.0781  | 33.6357 |
| 1750824904001  | 60.00s | 0.7434 | 0.3286  | 22.3784 |
| 1750825558261  | 44.15s | 0.9409 | 0.2095  | 30.6582 |
| 1750846199351  | 40.35s | 0.8269 | 0.2475  | 27.8340 |
| 1751090600427  | 58.16s | 0.9592 | 0.1248  | 32.1798 |
| **AVERAGE**    | 51.77s | 0.8792 | 0.2190  | 28.1669 |










