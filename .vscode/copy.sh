

cases=("banana_1" "blue_button" "blue_plastic_cube" "carambola" "cat_toys_2" "mangosteen"
"paw" "pink_button" "red_marker_pen_1" "shelves" "tomato" "yellow_brush" "avocado" "blanket"
"blue_marker_pen" "blue_plate" "carrots" "corn_1" "green_button" "onion_1" "pink_big_box" "screwdriver" "tomato_3")
for i in "${!cases[@]}"; do
    input=${cases[$i]}
    root_dir=/home/zzy/data/scan/$input
    echo $root_dir
    tosutil cp -r $root_dir tos://dexmal-fa-zzy/data/scan/
    rm -r $root_dir
done