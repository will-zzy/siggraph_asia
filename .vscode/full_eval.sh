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

NAME=3DV_CASIA # 请确保与full_train.sh中的NAME一致
n_offsets=10


DATA_PATH=/root/data/siga_round2
EVALUATE_DIR=/root/data/siga_round2/3DV_CASIA
LOG_FILE=$EVALUATE_DIR/metrics.json
READ_TIME_FILE=$EVALUATE_DIR/metrics_train.json
rm -f "$LOG_FILE"
for case in "${CASES[@]}"; do
    root_dir=$DATA_PATH/$case
    model_dir=$EVALUATE_DIR/$case
    ply_path=$(find "$model_dir/point_cloud" -mindepth 1 -maxdepth 1 -type d -print -quit)

    python \
        evaluate.py \
        -s $root_dir -m $model_dir -r 2 \
        --images images_gt_downsampled \
        --log_file $LOG_FILE \
        --read_time_file $READ_TIME_FILE \
        --antialiasing
done