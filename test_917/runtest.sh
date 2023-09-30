

#!/usr/bin/env bash
set -eu

LIST_DIR="output_930"
mkdir -p $LIST_DIR


for i in $(seq 365 501); do
    test_dir="../test_930/data_jiahan/enh-${i}.wav"
    output_file="$LIST_DIR/output_${i}.wav"
    python main.py --test_dir $test_dir --output_dir $output_file --ckpt ./models_experiment_1/experiment_1model_09_82.118638.h5
done
