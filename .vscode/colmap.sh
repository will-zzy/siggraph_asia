cuda_device=0
sparse_reconstruction(){
    mkdir -p $1/sparse
    colmap feature_extractor \
        --ImageReader.camera_model PINHOLE \
        --database_path $1/database.db \
        --image_path $1/$2 \
        --ImageReader.single_camera 1 \
        --SiftExtraction.peak_threshold 0.00666666671 #0.0066666666666666671
        # --SiftExtraction.max_image_size

    colmap exhaustive_matcher \
        --database_path $1/database.db \
        --FeatureMatching.use_gpu 1 

    # glomap mapper \
    #     --database_path $1/database.db \
    #     --image_path $1/$2 \
    #     --output_path $1/sparse \
    #     --Thresholds.min_inlier_num 30 # 30
    colmap mapper \
        --database_path $1/database.db \
        --image_path $1/$2 \
        --output_path $1/sparse \
        --Mapper.abs_pose_min_num_inliers 30 # 30
}
# mono_prior(){
#     cd ./metric3D
#     CUDA_VISIBLE_DEVICES=$cuda_device python \
#         mono/tools/densifyGaussianInitPoints.py \
#         mono/configs/HourglassDecoder/vit.raft5.giant2.py \
#         --load-from ./weight/metric_depth_vit_giant2_800k.pth \
#         --test_data_path $1/images \
#         --launcher None \
#         --vis true \
#         --inv_depth true
       
#         echo "mono prior done !"
# }
mono_prior(){
    cd ./metric3D
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$cuda_device python \
        mono/tools/densifyGaussianInitPoints.py \
        mono/configs/HourglassDecoder/vit.raft5.giant2.py \
        --load-from ./weight/metric_depth_vit_giant2_800k.pth \
        --root $1 \
        --k-views 8 \
        --samples-per-view 3000 \
        --strategy bg \
        --bg-percentile 0.9
    # -m debugpy --wait-for-client --listen localhost:5684 \
       
        echo "mono prior done !"
}

undistortion(){
    python \
        -m debugpy --wait-for-client --listen localhost:5684 \
        scripts/undistorted.py \
        --case $1 \
        --images-dir images_ori \
        --center-principal-point \
        --cameras-txt inputs/slam/cameras.txt 
}
preprocess(){
    rm -r $1/sparse
    python \
        ./scripts/preprocess.py \
        --root $1 \
        --video-name $2\_flip.mp4 \
        --videoinfo-txt inputs/videoInfo.txt \
        --out-images-dir images
    # -m debugpy --wait-for-client --listen localhost:5684 \
    # -m debugpy --wait-for-client --listen localhost:5684 \
}

addGaussianInitPoints(){

    python scripts/colmap_add_init_point.py \
      --root $1 \
      --images-dir images \
      --images-txt inputs/slam/images.txt \
      --cameras-txt inputs/slam/cameras.txt \
      --pair-stride 1 \
      --max-pairs 80 \
      --reproj-th 10.0 \
      --voxel 0.00




}

case=$1
root_dir=~/data/sa/$case

preprocess $root_dir $case
# sparse_reconstruction $root_dir images
mono_prior $root_dir
# addGaussianInitPoints $root_dir