# case=Epi7OJfbZDOd
# case=SdPeqQe2ESOd
# case=m57ePYzRxM49
case=3dgs

CUDA_VISIBLE_DEVICES=6 python \
    mono/tools/test_scale_cano.py \
    mono/configs/HourglassDecoder/vit.raft5.giant2.py \
    --load-from ./weight/metric_depth_vit_giant2_800k.pth \
    --test_data_path /home/zzy_group/data/comp/self/$case/images \
    --show-dir /home/zzy_group/data/comp/self/$case/mono_output \
    --launcher None