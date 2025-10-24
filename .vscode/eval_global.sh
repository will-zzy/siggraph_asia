
case=1747834320424
# case=1748165890960
root_dir=/home/zzy/data/sa/$case
# name=sig_GS_R2_speedy_forward
name=sig_GS_useAnySplatAnchor_global_transform_debug
n_offsets=10
model_dir=$root_dir/exp/$name

ply_path=/home/zzy/data/sa/1747834320424/exp/sig_GS_useAnySplatAnchor_global_transform_debug/point_cloud/iteration_6000
python \
    -m debugpy --wait-for-client --listen localhost:5686 \
    transformScaffold2Gaussian.py \
    -s $root_dir -m $model_dir -r 2 \
    --load_ply $ply_path \
    --n_offsets $n_offsets