# CONCORD: Efficient Clone-aware Pre-training for Source Code

This repository provides the code, data, and pre-trained models for the under-review paper: "CONCORD: Efficient Clone-aware Pre-training for Source Code".

## Getting Started

We provide two options to use our tool:
1. [Load our pre-trained model and fine-tune the model for downstream tasks](https://github.com/Anonymous-authors-2020/CONCORD#use-pre-trained-concord-for-fine-tuning)
2. [Pre-train the model from scratch](https://github.com/Anonymous-authors-2020/CONCORD#pre-train-concord-from-scatch)

### Data and Pre-trained Models
To quickly start, you need to first download our pre-training and fine-tuning data, together with the pre-trained models from the following link
  - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6518078.svg)](https://doi.org/10.5281/zenodo.6518078)

After downloading it, you need to unzip it, and you will see:
  - Data: `CONCORD_under_submission/data`
  - Pre-trained Models: `CONCORD_under_submission/pretrained_models`

## Use Pre-trained CONCORD for Fine-tuning
We provide the instuctions about how to fine-tune CONCORD for downstream tasks that we mentioned in the paper. We provide an example script for one benchmark in each task, and to run all benchmarks, please change the file names.
#### Semantic Clone Detection
```
python run_finetune_cc.py \
    --task_name poj104 \
    --model_name_or_path ../CONCORD_under_submission/pretrained_models/CONCORD/ \
    --config_name pretrain/config/concord_finetune_config.json \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model map \
    --evaluation_strategy steps \
    --max_seq_length 512 \
    --learning_rate 8e-6 \
    --num_train_epochs 10 \
    --output_dir poj104_cc_output \
    --cache_dir cache_dir \
    --tokenizer_name=vocab/multilingual_50k_vocab.txt \
    --save_steps=1000 \
    --logging_steps=1000 \
    --eval_steps 1000 \
    --save_total_limit=1 \
    --seed 42 \
    --fp16 \
    --warmup_ratio 0.1 \
    --train_file ../CONCORD_under_submission/data/finetune/poj_cc/train.jsonl \
    --validation_file ../CONCORD_under_submission/data/finetune/poj_cc/valid.jsonl \
    --test_file ../CONCORD_under_submission/data/finetune/poj_cc/test.jsonl \
    --overwrite_output_dir 2>&1 | tee poj104_cc_output/log_finetune
```

#### Bug Detection
```
python run_finetune_vd.py \
	--task_name cxg_vd \
	--model_name_or_path ../CONCORD_under_submission/pretrained_models/CONCORD/ \
	--config_name pretrain/config/concord_finetune_config.json \
	--per_device_eval_batch_size 8 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 1 \
	--do_train \
	--do_eval \
	--do_predict \
	--load_best_model_at_end \
	--metric_for_best_model acc \
	--evaluation_strategy steps \
	--max_seq_length 512 \
	--learning_rate 8e-6 \
	--num_train_epochs 5 \
	--output_dir d2a_vd_output \
	--cache_dir cache_dir \
	--tokenizer_name=vocab/multilingual_50k_vocab.txt \
	--save_total_limit=1 \
	--seed 42 \
	--fp16 \
	--warmup_ratio 0.05 \
	--eval_steps 100 \
	--save_steps=100 \
	--logging_steps=100 \
	--train_file ../CONCORD_under_submission/data/finetune/d2a_vd/train_func.csv \
	--validation_file ../CONCORD_under_submission/data/finetune/d2a_vd/valid_func.csv \
	--test_file ../CONCORD_under_submission/data/finetune/d2a_vd/test_func.csv \
	--overwrite_output_dir 2>&1 | tee $OUTPUT_DIR/log_finetune
```

#### Code Search
```
python run_finetune_cs_cxg.py \
    --model_name_or_path ../CONCORD_under_submission/pretrained_models/CONCORD-csnet/ \
    --config_name pretrain/config/concord_finetune_config.json \
    --eval_batch_size 96 \
    --train_batch_size 96 \
    --do_train \
    --do_eval \
    --do_test \
    --code_length 256 \
    --nl_length 128 \
    --learning_rate 8e-6 \
    --output_dir cxg_cs_output \
    --tokenizer_name=vocab/multilingual_50k_vocab.txt \
    --seed 42 \
    --warmup_ratio 0.01 \
    --num_train_epochs 100 \
    --train_data_file ../CONCORD_under_submission/data/finetune/cxg_codesearch/train.json \
    --eval_data_file ../CONCORD_under_submission/data/finetune/cxg_codesearch/valid.json \
    --test_data_file ../CONCORD_under_submission/data/finetune/cxg_codesearch/test.json \
    --codebase_file ../CONCORD_under_submission/data/finetune/cxg_codesearch/codebase.json 2>&1 | tee $OUTPUT_DIR/log_finetune
```

## Pre-train CONCORD from scratch
#### Phase-I Pre-training
1. Merge files in `../CONCORD_under_submission/data/pre-train/github/` into one single file `train.txt` or `test.txt`
2. BPE-tokenize the file using `vocab/multilingual_50k_vocab.model`
3. Split the long samples into 512 tokens sub-sample.
4. Run the following script. The batch size should be in total 2048, as we use 2 GPUs, so we set the batch size for each device to be 16 and accumulation steps to be 64.
```
python run_mlm.py \
    --preprocessing_num_workers 16 \
    --config_name pretrain/config/bert_MLM_config.json \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 64 \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --learning_rate 5e-4 \
    --max_steps 40000 \
    --output_dir phase_i_pretrain_output \
    --cache_dir $CACHE_DIR \
    --tokenizer_name=vocab/multilingual_50k_vocab.txt \
    --save_steps=10000 \
    --logging_steps=500 \
    --save_total_limit=1 \
    --seed 42 \
    --mlm_probability 0.15 \
    --line_by_line \
    --fp16 \
    --warmup_ratio 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --train_file train_bpe_512.txt \
    --validation_file valid_bpe_512.txt \
    --overwrite_output_dir 2>&1 | tee phase_i_pretrain_output/log.txt
```
#### Phase-II Pre-training
#### Code-only
1. Augment the samples in `../CONCORD_under_submission/data/pre-train/github/` using the this script
2. Run the following script. The batch size should be in total 512, as we use 2 GPUs, so we set the batch size for each device to be 4 and accumulation steps to be 64.
```
python run_concord_pretrain.py \
  --preprocessing_num_workers 4 \
	--config_name pretrain/config/concord_pretrain_config.json \
	--model_name_or_path phase_i_pretrain_output \
	--per_device_eval_batch_size 4 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 64 \
	--do_train \
	--do_eval \
	--evaluation_strategy steps \
	--eval_steps 3000 \
	--max_seq_length 512 \
	--learning_rate 5e-5 \
	--num_train_epochs 5 \
	--output_dir $OUTPUT_DIR \
	--cache_dir $CACHE_DIR \
	--tokenizer_name=vocab/multilingual_50k_vocab.txt \
	--save_steps=3000 \
	--logging_steps=500 \
	--save_total_limit=1 \
	--seed 42 \
	--mlm_probability 0.15 \
	--line_by_line \
	--fp16 \
	--warmup_ratio 0.01 \
	--train_file <augmented_train_file> \
	--validation_file <augmented_valid_file> \
	--overwrite_output_dir 2>&1 | tee $OUTPUT_DIR/log_pretrain.txt
```

#### Bi-modal CONCORD
1. Augment the samples in `../CONCORD_under_submission/data/pre-train/csnet/`.
2. Run the above script again with the new train/valid files. 

