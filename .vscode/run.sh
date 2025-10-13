
case=1748153841908
root_dir=/home/zzy/data/sa/$case
# name=sig_GS_R2_speedy_forward
name=sig_GS_R2_speedy_forward_LB_w_scaffold_withPoseOptimization_24000AddInit
# name=sig_GS_R2_baseline
model_dir=$root_dir/exp/$name
voxel_size=0.001
appearance_dim=16
n_offsets=4
update_init_factor=16
densify_until_iter=5000
update_from=500
update_until=$densify_until_iter
feat_dim=64
densify_grad_threshold=0.001
resolution=2



mlp_opacity_lr_init=0.002 # 0.002
mlp_cov_lr_init=0.004 # 0.004
mlp_color_lr_init=0.008 # 0.008
feature_lr=0.0075 # 0.0075

preprocess(){
    python \
        ./scripts/preprocess.py \
        --root $1 \
        --video-name $2\_flip.mp4 \
        --videoinfo-txt inputs/videoInfo.txt \
        --out-images-dir images
    # -m debugpy --wait-for-client --listen localhost:5684 \
    # -m debugpy --wait-for-client --listen localhost:5684 \


}


# preprocess $root_dir $case
bash .vscode/colmap.sh $case # 预处理脚本
rm -r $model_dir/test
python \
    train_dash.py -s \
    $root_dir -m $model_dir -r $resolution \
    --densify_mode freq \
    --resolution_mode const \
    --disable_viewer --eval \
    --antialiasing \
    --optimizer_type sparse_adam \
    --voxel_size $voxel_size \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim \
    --n_offsets $n_offsets \
    --densify_until_iter $densify_until_iter \
    --densify_grad_threshold $densify_grad_threshold \
    --feat_dim $feat_dim \
    --update_from $update_from \
    --update_until $update_until \
    --mlp_opacity_lr_init $mlp_opacity_lr_init \
    --mlp_cov_lr_init $mlp_cov_lr_init \
    --mlp_color_lr_init $mlp_color_lr_init \
    --feature_lr $feature_lr
    # -m debugpy --wait-for-client --listen localhost:5684 \
    # -m debugpy --wait-for-client --listen localhost:5684 \