# git需要更新秘钥时：
git config --global --replace-all user.password "要修改的密码"

# 与ROS环境分开，需要：
source ~/.bashrc
conda activate rlbench_rpt
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# 打开CoppeliaSim，可以修改任务和机械臂场景
bash /home/boxjod/Gym/RLBench/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/coppeliaSim.sh  

# 创建和修改测试RLBench任务【千万不要在这里的CoppeliaSim的界面里保存scence，要不会将任务设计场景覆盖到基础的task_design框架场景】
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5 # 可以
python3 RLBench/tools/task_builder_sawyer.py --task push_button # 可以
python3 RLBench/tools/task_builder_sawyer.py --task push_button # 可以
python3 RLBench/tools/task_builder_sawyer.py --task phone_on_base # 任务可以，但是有问题
python3 RLBench/tools/task_builder_sawyer.py --task basketball_in_hoop # 任务可以，但是有问题

python3 RLBench/tools/task_builder_sawyer.py --task basketball_in_hoop


# 演示数据集生成    
python3 RLBench/tools/dataset_generator_sawyer_act3.py \
    --save_path Datasets \
    --tasks basketball_in_hoop \
    --variations 1 \
    --episodes_per_task 50 \
; \

# 数据集可视化
python3 RPT_model/visualize_episodes.py --dataset_dir Datasets/push_button/variation0 --episode_idx 0
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
    --task_name push_button \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_episodes_sawyer4.py \
    --task_name push_button \
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
