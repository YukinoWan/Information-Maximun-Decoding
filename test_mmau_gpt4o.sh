#!/bin/bash

MMAU_DIR=data/MMAU

# iters=(1 2 3 4 5)
# for iter in ${iters[*]}; do
# MODEL_DIR="mispeech/r1-aqa"
# MODEL_DIR="Qwen/Qwen2-Audio-7B-Instruct"
MODEL_DIR="gpt-4o-20241120"
TEST_NAME="qwen2-5-caption-with-question-gpt4o-test"
OUT_DIR=evaluation/test_${TEST_NAME}


python3 src/test_mmau.py --model_path ${MODEL_DIR} --data_file caption/test_whisper-sampling/res_mmau_mini.json --audio_dir ${MMAU_DIR} --out_file ${OUT_DIR}/res_mmau_mini.json --batch_size 1  --mode "1-caption-evaluation" --temperature 0.7 || exit 1
python3 ${MMAU_DIR}/evaluation.py --input ${OUT_DIR}/res_mmau_mini.json > ${OUT_DIR}/eval_mmau_mini.txt || exit 1
    # python src/test_mmau.py --model_path ${MODEL_DIR} --data_file ${MMAU_DIR}/mmau-test.json --audio_dir ${MMAU_DIR} --out_file ${OUT_DIR}/res_mmau.json --batch_size 32 || exit 1
# done

# show Acc for each checkpoint (test-mini)
python3 src/utils/show_acc.py -i ${OUT_DIR} || exit 1
