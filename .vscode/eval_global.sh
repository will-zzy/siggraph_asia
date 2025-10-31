
case=1747834320424
# case=1748165890960
root_dir=/data1/zzy/sa/$case
# name=sig_GS_R2_speedy_forward
name=sig_GS_R2_speedy_forward_LB_w_scaffold_withPoseOptimization_24000AddInit
n_offsets=10
model_dir=$root_dir/exp/$name

ply_path=$model_dir/point_cloud/iteration_4635
python \
    evalGSReconstruction.py \
    -s $root_dir -m $model_dir -r 2 \
    --load_ply $ply_path \
    --n_offsets $n_offsets