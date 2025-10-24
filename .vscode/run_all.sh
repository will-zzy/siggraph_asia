# !/bin/bash
set -euo pipefail

# 定义所有case
CASES=(
    1747834320424
    1748153841908
    1748165890960
    1748242779841
    1748243104741
    1748265156579
    1748411766587
    1748422612463
    1748686118256
    1748689420856
    1748748200211
    1748781144981
    1748833935627
    1749369580718
    1749449291156
    1749606908096
    1749803955124
    1749864076665
    1749972078570
    1749974151642
    1749975239137
    1749977115648
    1750342099701
    1750342304509
    1750343446362
    1750383597053
    1750578027423
    1750824904001
    1750825558261
    1750846199351
    1751090600427
)

# 基础路径
ROOT_BASE="/home/jyxc/projects/siggraph_asia/data"  # 修改为你的数据根目录
NAME="sig_GS_useAnySplatAnchor_12_10"  # 修改为你的实验名称
VGGY_PATH=vggt  # 修改为你的VGGT模型路径


# 超参数设置
voxel_size=0.001
appearance_dim=16
update_init_factor=16
densify_grad_threshold=0.0005

feat_dim=64
n_offsets=10 # 每个anchor的子高斯数

densify_until_iter=5000
update_from=100
densify_from_iter=$update_from
update_until=$densify_until_iter
densification_interval=300
FF_downsample=128 # 对anySplat的点下采样倍数，用于充当anchor

# 预处理
preprocess(){
    python \
        ./scripts/preprocess.py \
        --root $1 \
        --video-name $2\_flip.mp4 \
        --videoinfo-txt inputs/videoInfo.txt \
        --out-images-dir images
}

# 创建文件夹存放所有日志
EVAL_DIR="$(pwd)/eval_logs/anysplat_scaffold/" 
mkdir -p "$EVAL_DIR"
LOG_FILE="$EVAL_DIR/all_cases_results.csv"

# 初始化日志文件存则则删除旧文件
rm -f "$LOG_FILE"
echo "case,time,SSIM,LPIPS,PSNR" > "$LOG_FILE"

# 遍历所有case
for case in "${CASES[@]}"; do
    root_dir="$ROOT_BASE/$case"
    model_dir="$root_dir/exp/$NAME"

    # 如果 sparse 文件夹存在则跳过预处理
    if [ -d "$root_dir/sparse" ]; then
        echo "Skipping preprocessing for case: $case (sparse directory exists)" 
    else
        echo "Preprocessing case: $case"
        preprocess "$root_dir" "$case"
    fi

    # 删除旧的测试结果文件夹
    if [ -d "$model_dir/test" ]; then
        rm -r "$model_dir/test"
    fi
    if [ -d "$model_dir/point_cloud" ]; then
        rm -r "$model_dir/point_cloud"
    fi

    ANY_SPLAT_VGGT_WEIGHTS=$VGGT_PATH python train_dash.py -s \
        $root_dir -m $model_dir -r 2 \
        --resolution_mode const \
        --densify_until_iter $densify_until_iter \
        --densify_mode freq \
        --disable_viewer \
        --antialiasing \
        --optimizer_type sparse_adam \
        --voxel_size $voxel_size \
        --update_init_factor $update_init_factor \
        --appearance_dim $appearance_dim \
        --densify_grad_threshold $densify_grad_threshold \
        --densify_until_iter $densify_until_iter \
        --densify_from_iter $densify_from_iter \
        --densification_interval $densification_interval \
        --update_from $update_from \
        --update_until $update_until \
        --n_offsets $n_offsets \
        --FF_downsample $FF_downsample \
        --feat_dim $feat_dim \
        --iterations 6000 \
        --log_file "$LOG_FILE"

    sleep 2  # 确保日志文件写入完成
done

echo ""
echo "----------------------------------------"
echo ""

# 对所有case的结果进行处理，汇总并计算平均值，输出到 txt 并在终端打印
SUMMARY_FILE="$EVAL_DIR/summary.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

rm -f "$SUMMARY_FILE"

# 读取 CSV，打印每个 case 并计算各列平均值，结果同时写入 SUMMARY_FILE 并在终端显示
awk -F, '
NR==1 { next } # skip header
{
    # trim spaces for all fields
    for(i=1;i<=NF;i++) { gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    c=$1
    time_raw=$2
    s=$3+0
    l=$4+0
    p=$5+0

    # extract numeric seconds from time (e.g. "30s" -> 30)
    tmp=time_raw
    gsub(/[^0-9.]/, "", tmp)
    tnum = (tmp=="" ? 0 : tmp+0)

    idx = ++n
    cases[idx]=c
    times[idx]=time_raw
    times_num[idx]=tnum
    ssim[idx]=s
    lpips[idx]=l
    psnr[idx]=p

    sum_time += tnum
    sum_ssim += s
    sum_lpips += l
    sum_psnr += p
    count++
}
END {
    # per-case listing
    printf "%-24s %10s %10s %10s %10s\n", "case", "time", "SSIM", "LPIPS", "PSNR"
    for(i=1;i<=n;i++) {
        printf "%-24s %10.2fs %10.4f %10.4f %10.4f\n", cases[i], times[i], ssim[i], lpips[i], psnr[i]
    }

    # averages across all entries
    if (count>0) {
        avg_time = sum_time / count
        printf "\n%-24s %10.2fs %10.4f %10.4f %10.4f\n", "AVERAGE", avg_time, sum_ssim/count, sum_lpips/count, sum_psnr/count
    } else {
        print "\nNo data to average."
    }
}
' "$LOG_FILE" | tee "$SUMMARY_FILE"

echo ""
echo "----------------------------------------"
echo ""
echo "Logs written to $LOG_FILE"
echo "Summary written to $SUMMARY_FILE"

