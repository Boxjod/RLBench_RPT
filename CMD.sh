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
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5
python3 RLBench/tools/task_builder_sawyer.py --task push_button
python3 RLBench/tools/task_builder_sawyer.py --task push_buttons


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

# 无法训练
# 更新conda 
# 当前 conda 23.5.0
# 更新到 conda 24.5.0

# 能用的xj_rlbench, pytorch=2.1.2
nvidia-cublas-cu12            12.1.3.1
nvidia-cuda-cupti-cu12        12.1.105
nvidia-cuda-nvrtc-cu12        12.1.105
nvidia-cuda-runtime-cu12      12.1.105
nvidia-cudnn-cu12             8.9.2.26
nvidia-cufft-cu12             11.0.2.54
nvidia-curand-cu12            10.3.2.106
nvidia-cusolver-cu12          11.4.5.107
nvidia-cusparse-cu12          12.1.0.106
nvidia-ml-py                  12.555.43
nvidia-nccl-cu12              2.18.1 #############
nvidia-nvjitlink-cu12         12.5.40 #############
nvidia-nvtx-cu12              12.1.105

# 能用的 服务器pytorch=2.3.1
nvidia-cublas-cu12            12.1.3.1
nvidia-cuda-cupti-cu12        12.1.105
nvidia-cuda-nvrtc-cu12        12.1.105
nvidia-cuda-runtime-cu12      12.1.105
nvidia-cudnn-cu12             8.9.2.26
nvidia-cufft-cu12             11.0.2.54
nvidia-curand-cu12            10.3.2.106
nvidia-cusolver-cu12          11.4.5.107
nvidia-cusparse-cu12          12.1.0.106
nvidia-nccl-cu12              2.20.5 #############
nvidia-nvjitlink-cu12         12.5.82 #############
nvidia-nvtx-cu12              12.1.105

# 不能用的 pytorch=2.3.1
nvidia-cublas-cu12            12.1.3.1
nvidia-cuda-cupti-cu12        12.1.105
nvidia-cuda-nvrtc-cu12        12.1.105
nvidia-cuda-runtime-cu12      12.1.105
nvidia-cudnn-cu12             8.9.2.26
nvidia-cufft-cu12             11.0.2.54
nvidia-curand-cu12            10.3.2.106
nvidia-cusolver-cu12          11.4.5.107
nvidia-cusparse-cu12          12.1.0.106
nvidia-ml-py                  12.555.43
nvidia-nccl-cu12              2.20.5 #############
nvidia-nvjitlink-cu12         12.5.40 #############
nvidia-nvtx-cu12              12.1.105



conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia



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
