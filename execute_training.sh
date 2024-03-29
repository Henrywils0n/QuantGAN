#!/bin/bash

# Define lists of alpha_d and alpha_g values
alpha_d_list=(1 3 5 10 100)
alpha_g_list=(0.5 1 3 5 10 100)

trained_combinations=("1,0.5" "1,1")

for alpha_d in "${alpha_d_list[@]}"; do
    for alpha_g in "${alpha_g_list[@]}"; do
        if (( $(bc <<< "$alpha_d <= 1") )) && (( $(bc <<< "$alpha_g >= $alpha_d / ($alpha_d + 1)") )) && ! [[ "${trained_combinations[*]}" =~ "$alpha_d,$alpha_g" ]]; then
            echo "Executing with alpha_d=$alpha_d, alpha_g=$alpha_g"
            python -u tf_train.py --alpha_d "$alpha_d" --alpha_g "$alpha_g"
        elif (( $(bc <<< "$alpha_d > 1") )) && (( $(bc <<< "$alpha_g >= $alpha_d / 2") )) && (( $(bc <<< "$alpha_g <= $alpha_d") )) && ! [[ "${trained_combinations[*]}" =~ "$alpha_d,$alpha_g" ]]; then
            echo "Executing with alpha_d=$alpha_d, alpha_g=$alpha_g"
            python -u tf_train.py --alpha_d "$alpha_d" --alpha_g "$alpha_g"
        fi
    done
done