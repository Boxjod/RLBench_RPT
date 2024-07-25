#!/bin/bash

# 生成数据集
# python3 RLBench/tools/dataset_generator_sawyer_act3.py \
#     --save_path Datasets \
#     --tasks push_button \
#     --variations 1 \
#     --episodes_per_task 50

model_type=(ACT0E0 ACT3E0 ACT3E2 ACT3E3)
epoch_list=(8000 9000 10000 11000 12000 13000 14000)
backbone_list=("efficientnet_b0")
chunk_size=(20 30)
for model in ${model_type[@]}
  do
  for epoch in ${epoch_list[@]}
    do
    for backbone in ${backbone_list[@]}
      do
      for chunk in ${chunk_size[@]}
        do
        echo '##################################################################'
        echo 'train on model=' $model ', epoch=' $epoch ',  chunk_size='$chunk 
        echo '##################################################################'
        
        CUDA_VISIBLE_DEVICES=0 python3 RPT_model/imitate_episodes_sawyer4.py \
        --task_name push_button \
        --ckpt_dir sorting_program5 \
        --policy_class $model --kl_weight 10 --chunk_size $chunk --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
        --num_epochs $epoch  --lr 1e-5 --seed 0 --backbone $backbone \
        ; \
        CUDA_VISIBLE_DEVICES=0 python3 RPT_model/imitate_episodes_sawyer4.py \
        --task_name push_button \
        --ckpt_dir sorting_program5 \
        --policy_class $model --kl_weight 10 --chunk_size $chunk --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
        --num_epochs $epoch  --lr 1e-5 --seed 0 --backbone $backbone \
        --eval --temporal_agg 
        done
      done
    done
  done
  
  

