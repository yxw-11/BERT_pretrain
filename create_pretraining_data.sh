#!/bin/bash

#RANK=0
#WORLD_SIZE=1
#DATA_PATH=<Specify path and file prefix>_text_sentence
#CHECKPOINT_PATH="/Users/sirerix/Documents/JHU 2021 Summer-Capstone/Megatron-LM-main/tmp/bert_uncased_L-12_H-768_A-12/pytorch_model.bin"
#SAVE_PATH="/Users/sirerix/Documents/JHU 2021 Summer-Capstone/Megatron-LM-main/tmp/new_ckeckpt.bin"

# chmod u+r+x erix_created/create_pretraining_data.sh
# erix_created/create_pretraining_data.sh


cd "$DOCS"
## use * to replace
export BERT_BASE_DIR="/Users/sirerix/Documents/JHU 2021 Summer-Capstone/Megatron-LM-main/tmp/bert_uncased_L-12_H-768_A-12"
python erix_created/create_pretraining_data.py \
  --input_file=./tmp/sententrced_text.txt \
  --output_file=tmp/training_data_text.tfrecord\
  --vocab_file="$BERT_BASE_DIR/vocab.txt" \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5