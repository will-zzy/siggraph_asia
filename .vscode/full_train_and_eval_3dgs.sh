

# case=Truck

CASES=(
    1747834320424
    1748153841908
    1748165890960
    1748242779841
    1748243104741
    1749449291156
    1749606908096
    1749803955124
    1750578027423
    1750824904001
    1750825558261
    1750846199351
    1751090600427
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
VGGT_PATH=$(pwd)/vggt
densify_grad_threshold=0.0002


feat_dim=64
n_offsets=10 # 每个anchor的子高斯数

densify_until_iter=5000
densify_from_iter=500
densification_interval=100
# FF_downsample=100000 # 对anySplat的点下采样倍数，用于充当anchor
FF_downsample=16 # 对anySplat的点下采样倍数，用于充当anchor


opacity_lr=0.02 # 0.02
scaling_lr=0.007 # 0.007
rotation_lr=0.002 # 0.002
feature_dc_lr=0.01 # 0.01
feature_rest_lr=0.0005 # 0.0005
xyz_lr_init=0.00016 # 0.00016
xyz_lr_final=0.000016 # 0.000016
max_n_gaussian=3000000
iterations=12000



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
        rm -rf "$model_dir/test"
        rm -rf "$model_dir/point_cloud"
    fi
    ANY_SPLAT_VGGT_WEIGHTS=$VGGT_PATH CUDA_VISIBLE_DEVICES=$device python \
        train_dash.py -s \
        $root_dir -m $model_dir -r 2 \
        --resolution_mode freq \
        --densify_until_iter $densify_until_iter \
        --densify_mode freq \
        --disable_viewer \
        --antialiasing \
        --optimizer_type sparse_adam \
        --densify_grad_threshold $densify_grad_threshold \
        --densify_until_iter $densify_until_iter \
        --densify_from_iter $densify_from_iter \
        --densification_interval $densification_interval \
        --densify_until_iter $densify_until_iter \
        --FF_downsample $FF_downsample \
        --feature_dc_lr $feature_dc_lr \
        --feature_rest_lr $feature_rest_lr \
        --xyz_lr_init $xyz_lr_init \
        --xyz_lr_final $xyz_lr_final \
        --scaling_lr $scaling_lr \
        --rotation_lr $rotation_lr \
        --opacity_lr $opacity_lr \
        --images images_gt_downsampled \
        --log_file $LOG_FILE \
        --max_n_gaussian $max_n_gaussian \
        --iterations $iterations \
        --useFF
        # --train_test_exp
        # --use_feat_bank true\
        # -m debugpy --wait-for-client --listen localhost:5685 \
        # -m debugpy --wait-for-client --listen localhost:5684 \

        # rm -r $model_dir/mono_depths
done

echo ""
echo "----------------------------------------"
echo ""
