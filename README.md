# CONCORD: Clone-aware Contrastive Learning for Source Code

This repository provides the code, data, and pre-trained models for the under-review paper: "CONCORD: Clone-aware Contrastive Learning for Source Code".

## Getting Started

We provide two options to use our tool:
1. Load our pre-trained model and fine-tune the model for downstream tasks
2. Pre-train the model from scratch

### Data and Pre-trained Models
To quickly start, you need to first download our pre-trained model and fine-tuning data from the following link
  - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7309076.svg)](https://doi.org/10.5281/zenodo.7309076)

## Use Pre-trained CONCORD for Fine-tuning
We provide the instuctions about how to fine-tune CONCORD for downstream tasks. We provide an example script for one benchmark in each task, and to run all benchmarks, please change the file names and arguments accordingly
### Semantic Clone Detection
```
python run_concord_finetune_cc_cxg.py \
        --task poj104 \
        --tokenizer_name vocab/vocab_50k.txt \
        --model_name_or_path $MODEL_PATH \
        --config_name config/concord_finetune_config.json \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --do_eval \
        --do_test \
        --block_size 512 \
        --learning_rate 8e-6 \
        --num_train_epochs 2 \
        --output_dir $OUTPUT_DIR \
        --cache_dir $CACHE_DIR \
        --save_steps=1000 \
        --seed 42 \
        --fp16 \
        --warmup_ratio 0.1 \
        --train_data_file $POJ104_TRAIN_FILE \
        --eval_data_file $POJ104_VALID_FILE \
        --test_data_file $POJ104_TEST_FILE \
        --overwrite_output_dir
```

### Bug Detection
```
python run_concord_finetune_vd.py \
	--task_name cxg_vd \
    	--tokenizer_name vocab/vocab_50k.txt \
	--model_name_or_path $MODEL_PATH \
	--config_name config/concord_finetune_config.json \
	--per_device_eval_batch_size 8 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 1 \
	--do_train \
	--do_eval \
	--load_best_model_at_end \
	--metric_for_best_model f1 \
	--evaluation_strategy steps \
	--eval_steps 400 \
	--max_seq_length 512 \
	--learning_rate 8e-6 \
	--num_train_epochs 10 \
	--output_dir $OUTPUT_DIR \
	--cache_dir $CACHE_DIR \
	--save_steps=400 \
	--logging_steps=400 \
	--save_total_limit=1 \
	--seed 42 \
	--fp16 \
	--train_file $RV_VD_TRAIN_FILE \
	--validation_file $RV_VD_VALID_FILE \
	--overwrite_output_dir
```

## Pre-train CONCORD from scratch
### Phase-I Pre-training
Please use `run_mlm.py`.
### Phase-II Pre-training
Please use `run_concord_pretrain.py`.
