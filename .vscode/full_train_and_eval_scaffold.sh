

# case=Truck

CASES=(
    # 1747834320424
    # 1748153841908
    1748165890960
    # 1748242779841
    # 1748243104741
    # 1749449291156
    # 1749606908096
    # 1749803955124
    # 1750578027423
    # 1750824904001
    # 1750825558261
    # 1750846199351
    # 1751090600427
)


mono_depth(){
    rm -rf $1/mono_depths
    python \
    $(pwd)/metric3D/mono/tools/test_scale_cano.py \
    $(pwd)/metric3D/mono/configs/HourglassDecoder/vit.raft5.giant2.py \
    --load-from $(pwd)/metric3D/weight/metric_depth_vit_giant2_800k.pth \
    --test_data_path $1 \
    --show-dir $1 \
    --output-dir $2
}


NAME=3DV_CASIA
BASE_DIR=/root/data/eval_data_pinhole
EXP_DIR=/root/data/eval_data_pinhole/$NAME
# case=1747834320424
# case=1748165890960
voxel_size=0.001
appearance_dim=16
update_init_factor=16
VGGT_PATH=$(pwd)/vggt
densify_grad_threshold=0.0005


feat_dim=64
n_offsets=10 # 每个anchor的子高斯数

densify_until_iter=4500
update_from=500
densify_from_iter=$update_from
update_until=$densify_until_iter
update_interval=100
# FF_downsample=100000 # 对anySplat的点下采样倍数，用于充当anchor
FF_downsample=80 # 对anySplat的点下采样倍数，用于充当anchor


MLP_OPACITY_LR_INIT=0.005 # 0.002
MLP_COV_LR_INIT=0.003 # 0.004
MLP_COLOR_LR_INIT=0.012 # 0.008
FEATURE_LR_INIT=0.01 # 0.0075
OFFSET_LR_INIT=0.01 # 0.01
MAX_N_GAUSSIAN=3000000



# preprocess(){
#     rm -r $1/sparse
#     python \
#         ./scripts/preprocess.py \
#         --root $1 \
#         --video-name $2\_flip.mp4 \
#         --videoinfo-txt inputs/videoInfo.txt \
#         --out-images-dir images
#     # -m debugpy --wait-for-client --listen localhost:5684 \
#     # -m debugpy --wait-for-client --listen localhost:5684 \
# }
# preprocess $root_dir $case
device=0
#  --eval
# rm -r $model_dir/test
LOG_FILE=$EXP_DIR/metrics.json
rm -f "$LOG_FILE"
for case in "${CASES[@]}"; do
    
    root_dir=$BASE_DIR/$case
    # name=sig_GS_R2_speedy_forward
    name=sig_GS_round2
    # name=sig_GS_R2_baseline
    model_dir=$EXP_DIR/$case
    # mono_depth $root_dir $model_dir

    if [ -d "$model_dir/test" ]; then
        rm -r "$model_dir/test"
    fi
    ANY_SPLAT_VGGT_WEIGHTS=$VGGT_PATH CUDA_VISIBLE_DEVICES=$device python \
        train_dash.py -s \
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
        --update_interval $update_interval \
        --update_from $update_from \
        --update_until $update_until \
        --n_offsets $n_offsets \
        --FF_downsample $FF_downsample \
        --feat_dim $feat_dim \
        --mlp_opacity_lr_init $MLP_OPACITY_LR_INIT \
        --mlp_cov_lr_init $MLP_COV_LR_INIT \
        --mlp_color_lr_init $MLP_COLOR_LR_INIT \
        --feature_lr $FEATURE_LR_INIT \
        --offset_lr_init $OFFSET_LR_INIT \
        --images images_gt_downsampled \
        --log_file $LOG_FILE \
        --max_n_gaussian $MAX_N_GAUSSIAN \
        --useFF
        # --use_feat_bank true\
        # -m debugpy --wait-for-client --listen localhost:5685 \
        # -m debugpy --wait-for-client --listen localhost:5684 \

        # rm -r $model_dir/mono_depths
done

echo ""
echo "----------------------------------------"
echo ""
