# 打开CoppeliaSim，可以修改任务和机械臂场景
bash ~/workspace/CoppeliaSim/coppeliaSim.sh  

# 创建和修改测试RLBench任务【千万不要在这里的CoppeliaSim的界面里保存scence，要不会将任务设计场景覆盖到基础的task_design框架场景】
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5

# 演示数据集生成    
python3 RLBench/tools/dataset_generator_sawyer_act3.py \
    --save_path Datasets \
    --tasks sorting_program5 \
    --variations 1 \
    --episodes_per_task 50 \
; \

# 数据集可视化
python3 RPT_model/visualize_episodes.py --dataset_dir Datasets/sorting_program5/variation0 --episode_idx 0
# 修改数据集
python3 RPT_model/mod_datasets.py

## train and eval
###########################################################################################################################
# 任务close_jar
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
    
    
######################################################
# 完整任务学习
python3 RLBench/tools/dataset_generator_sawyer_act3.py \
    --save_path Datasets \
    --tasks sorting_program5 \
    --variations 1 \
    --episodes_per_task 50 \
    ;\
    
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 4000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 4000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 6000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 6000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 7000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 7000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 8000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 8000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 9000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 9000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 10000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 10000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 11000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 11000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 12000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 12000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
