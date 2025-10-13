


# R2
# case=Ignatius     # speedy_forward = 173s , baseline = 181s | 20.48/21.64  22.61/27.26
# case=Truck        # speedy_forward = 159s , baseline = 173s | 21.95/22.27  25.95/27.71
# case=Barn         # speedy_forward = 180s , baseline = 191s | 24.63/24.31  29.13/30.04
# case=Meetingroom  # speedy_forward = 137s , baseline = 145s | 22.61/21.85  25.90/28.14
# case=Caterpillar  # speedy_forward = 171s , baseline = 183s | 21.21/24.17  24.03/29.36

# R4
# case=Ignatius     # speedy_forward = 132s | 20.78/21.84  23.24/29.24
# case=Truck        # speedy_forward = 129s | 22.13/22.57  26.77/28.84
# case=Barn         # speedy_forward = 141s | 24.70/24.55  30.27/31.86
# case=Meetingroom  # speedy_forward = 116s | 21.99/20.69  26.00/28.63
# case=Caterpillar  # speedy_forward = 143s | 21.42/24.38  24.79/30.92


root_dir=/home/zzy/data/public_data/TrainingSet/$case
# name=sig_GS_R2_speedy_forward
name=sig_GS_R4_speedy_forward
# name=sig_GS_R2_baseline
model_dir=$root_dir/exp/$name

# cases=("Ignatius" "Truck" "Barn" "Meetingroom" "Caterpillar")
cases=("Barn" "Meetingroom" "Caterpillar")
for i in "${!cases[@]}"; do
    c=${cases[$i]}
    root_dir=/home/zzy/data/public_data/TrainingSet/$c
    model_dir=$root_dir/exp/$name
    python \
        train_dash.py -s \
        $root_dir -m $model_dir -r 4 \
        --densify_mode freq \
        --resolution_mode freq \
        --densify_until_iter 27000 \
        --disable_viewer --eval \
        --antialiasing \
        --optimizer_type sparse_adam
        # --quiet
done