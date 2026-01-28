

# case=Truck

CASES=(
    1747834320424
    # 1748153841908
    # 1748165890960
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

BASE_DIR=/root/data/siga_round2
EXP_DIR=/root/data/siga_round2/3DV_CASIA

VGGT_PATH=$(pwd)/vggt

densify_grad_threshold=0.0002
densify_until_iter=5000
densify_from_iter=500
densification_interval=100

FF_downsample=16 # The downsampling factor for the points of anySplat


opacity_lr=0.01 # 0.02
scaling_lr=0.005 # 0.007
rotation_lr=0.001 # 0.002
feature_dc_lr=0.01 # 0.01
feature_rest_lr=0.0005 # 0.0005
xyz_lr_init=0.00016 # 0.00016
xyz_lr_final=0.000016 # 0.000016
max_n_gaussian=3000000
iterations=12000

device=0

LOG_FILE=$EXP_DIR/metrics_train.json
rm -f "$LOG_FILE"

for case in "${CASES[@]}"; do
    
    root_dir=$BASE_DIR/$case
    model_dir=$EXP_DIR/$case
    mono_depth $root_dir $model_dir # comment it after the first time

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
        --use_pose_optimization
        # --useFF
        # -m debugpy --wait-for-client --listen localhost:5685 \

        # rm -r $model_dir/mono_depths
done

echo ""
echo "----------------------------------------"
echo ""
