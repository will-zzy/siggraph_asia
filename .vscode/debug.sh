

case=Truck
root_dir=/home/zzy/data/public_data/TrainingSet/$case
# name=sig_GS_R2_speedy_forward
name=sig_GS_R2_scaffold_debug_freq
# name=sig_GS_R2_baseline
model_dir=$root_dir/exp/$name
voxel_size=0.01
appearance_dim=16
update_init_factor=16

python \
    train_dash.py -s \
    $root_dir -m $model_dir -r 2 \
    --resolution_mode const \
    --densify_until_iter 15000 \
    --densify_mode freq \
    --disable_viewer --eval \
    --antialiasing \
    --optimizer_type sparse_adam \
    --voxel_size $voxel_size \
    --update_init_factor $update_init_factor \
    --appearance_dim $appearance_dim 
    # --use_feat_bank true\
    # -m debugpy --wait-for-client --listen localhost:5684 \