# !/bin/bash
set -euo pipefail

# # 定义所有case
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

# CASES=(
#     1747834320424
#     1750825558261
#     1750846199351
#     1751090600427
# )
# 基础路径
ROOT_BASE="/data1/zzy/sa"
NAME="sig_GS_R2_speedy_forward_LB_w_scaffold"
VGGY_PATH=/home/zzy/engineer/siggraph_asia/vggt
# 超参数设置
VOXEL_SIZE=0.001
APPEARANCE_DIM=32
N_OFFSETS=10
UPDATE_INIT_FACTOR=16
DENSIFY_UNTIL_ITER=4000
UPDATE_FROM=500
UPDATE_UNTIL=${DENSIFY_UNTIL_ITER}
FEAT_DIM=64
DENSIFY_GRAD_THRESHOLD=0.0005
RESOLUTION=2
MLP_OPACITY_LR_INIT=0.002 # 0.002
MLP_COV_LR_INIT=0.004 # 0.004
MLP_COLOR_LR_INIT=0.014 # 0.008
OFFSET_LR_INIT=0.005 # 0.01
FEATURE_LR=0.0075 # 0.0075

FF_downsample=128 # 对anySplat的点下采样倍数，用于充当anchor
densification_interval=300

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
EVAL_DIR="$(pwd)/eval_logs" 
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

    # Metric3D初始化
    # bash .vscode/colmap.sh $case
    preprocess "$root_dir" "$case"

    # 删除旧的测试结果文件夹
    if [ -d "$model_dir/test" ]; then
        rm -r "$model_dir/test"
    fi

    ANY_SPLAT_VGGT_WEIGHTS=$VGGY_PATH python \
        train_dash.py -s "$root_dir" -m "$model_dir" -r $RESOLUTION \
        --densify_mode freq \
        --resolution_mode const \
        --disable_viewer --eval \
        --antialiasing \
        --optimizer_type sparse_adam \
        --voxel_size $VOXEL_SIZE \
        --update_init_factor $UPDATE_INIT_FACTOR \
        --appearance_dim $APPEARANCE_DIM \
        --n_offsets $N_OFFSETS \
        --densify_until_iter $DENSIFY_UNTIL_ITER \
        --densify_grad_threshold $DENSIFY_GRAD_THRESHOLD \
        --densification_interval $densification_interval \
        --FF_downsample $FF_downsample \
        --feat_dim $FEAT_DIM \
        --update_from $UPDATE_FROM \
        --update_until $UPDATE_UNTIL \
        --mlp_opacity_lr_init $MLP_OPACITY_LR_INIT \
        --mlp_cov_lr_init $MLP_COV_LR_INIT \
        --mlp_color_lr_init $MLP_COLOR_LR_INIT \
        --feature_lr $FEATURE_LR \
        --log_file "$LOG_FILE" \
        --offset_lr_init $OFFSET_LR_INIT
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
NF>=5 {
    # trim spaces
    for(i=1;i<=5;i++) { gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    c=$1
    # normalize time to "30s" or "60s" if possible
    if ($2 ~ /30/) t="30s"
    else if ($2 ~ /60/) t="60s"
    else t=$2

    ssim[c,t]=$3+0
    lpips[c,t]=$4+0
    psnr[c,t]=$5+0

    if (!(c in seen)) { cases[++ncases]=c; seen[c]=1 }
    # accumulate sums and counts per time
    sum_ssim[t]+=ssim[c,t]
    sum_lpips[t]+=lpips[c,t]
    sum_psnr[t]+=psnr[c,t]
    count[t]++
}
END {
    # header
    printf "%-24s %10s %10s %10s   %10s %10s %10s\n", "case", "30s_PSNR", "30s_SSIM", "30s_LPIPS", "60s_PSNR", "60s_SSIM", "60s_LPIPS"
    for (i=1;i<=ncases;i++) {
        c=cases[i]
        has30 = ((c, "30s") in psnr)
        has60 = ((c, "60s") in psnr)
        if (has30) { p30=psnr[c,"30s"]; s30=ssim[c,"30s"]; l30=lpips[c,"30s"] }
        if (has60) { p60=psnr[c,"60s"]; s60=ssim[c,"60s"]; l60=lpips[c,"60s"] }

        if (has30 && has60) {
            printf "%-24s %10.4f %10.4f %10.4f   %10.4f %10.4f %10.4f\n", c, p30, s30, l30, p60, s60, l60
        } else if (has30 && !has60) {
            printf "%-24s %10.4f %10.4f %10.4f   %10s %10s %10s\n", c, p30, s30, l30, "NA","NA","NA"
        } else if (!has30 && has60) {
            printf "%-24s %10s %10s %10s   %10.4f %10.4f %10.4f\n", c, "NA","NA","NA", p60, s60, l60
        } else {
            printf "%-24s %10s %10s %10s   %10s %10s %10s\n", c, "NA","NA","NA", "NA","NA","NA"
        }
    }

    print ""
    # averages for 30s
    if (count["30s"]>0) {
        printf "%-24s %10.4f %10.4f %10.4f\n", "AVERAGE(30s)", sum_psnr["30s"]/count["30s"], sum_ssim["30s"]/count["30s"], sum_lpips["30s"]/count["30s"]
    } else {
        print "No valid 30s data to average."
    }
    # averages for 60s
    if (count["60s"]>0) {
        printf "%-24s %10.4f %10.4f %10.4f\n", "AVERAGE(60s)", sum_psnr["60s"]/count["60s"], sum_ssim["60s"]/count["60s"], sum_lpips["60s"]/count["60s"]
    } else {
        print "No valid 60s data to average."
    }
}
' "$LOG_FILE" | tee "$SUMMARY_FILE"

echo ""
echo "----------------------------------------"
echo ""
echo "Logs written to $LOG_FILE"
echo "Summary written to $SUMMARY_FILE"

