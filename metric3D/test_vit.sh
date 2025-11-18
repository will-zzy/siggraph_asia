

case=Epi7OJfbZDOd
root_dir=/home/zzy_group/data/comp/self/$case
# case=m57ePYzRxM49
# case=SdPeqQe2ESOd
# case=3dgs

python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/vit.raft5.giant2.py' \
    --load-from ./weight/metric_depth_vit_giant2_800k.pth \
    --show-dir $root_dir/mono_output \
    --test_data_path $root_dir \
    --launcher None