

# case=Truck
case=1747834320424
# case=1748165890960
root_dir=/home/zzy/data/sa/$case
# name=sig_GS_R2_speedy_forward
name=sig_GS_useAnySplatAnchor_12_10
# name=sig_GS_R2_baseline
model_dir=$root_dir/exp/$name
voxel_size=0.001
appearance_dim=16
update_init_factor=16
VGGY_PATH=/home/zzy/lib/siggraph_asia/vggt
densify_grad_threshold=0.001



feat_dim=64
n_offsets=10 # 每个anchor的子高斯数

densify_until_iter=5000
update_from=100
densify_from_iter=$update_from
update_until=$densify_until_iter
densification_interval=300
FF_downsample=16 # 对anySplat的点下采样倍数，用于充当anchor


#  --eval
rm -r $model_dir/test
ANY_SPLAT_VGGT_WEIGHTS=$VGGY_PATH python \
    -m debugpy --wait-for-client --listen localhost:5684 \
    train_dash.py -s \
    $root_dir -m $model_dir -r 2 \
    --resolution_mode const \
    --densify_until_iter 15000 \
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
    --feat_dim $feat_dim
    # --useFF
    # --use_feat_bank true\