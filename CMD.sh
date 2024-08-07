# train on the server
conda activate rlbench_rpt
cd ~/workspace/RLBench_RPT
. do_server.sh

# git key
git config --global --replace-all user.password "要修改的密码"

# separate with ROS
source ~/.bashrc
conda activate rlbench_rpt
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# open CoppeliaSim，change the scence 
bash /home/boxjod/Gym/RLBench/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/coppeliaSim.sh  

# 创建和修改测试RLBench任务【千万不要在这里的CoppeliaSim的界面里保存scence，要不会将任务设计场景覆盖到基础的task_design框架场景】
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5 
python3 RLBench/tools/task_builder_sawyer.py --task push_button
python3 RLBench/tools/task_builder_sawyer.py --task basketball_in_hoop 


python3 RLBench/tools/task_builder_sawyer.py --task phone_on_base # 可以但，效果很差很差 ×××××××××
python3 RLBench/tools/task_builder_sawyer.py --task lamp_on   # 可以，但效果比较均一 ×××××××××××××
python3 RLBench/tools/task_builder_sawyer.py --task lift_numbered_block # 可以，但有任务重复，视觉要求很高 ××××××××××××××
python3 RLBench/tools/task_builder_sawyer.py --task light_bulb_out # 可以，很有难度 ××××××××××××××

python3 RLBench/tools/task_builder_sawyer.py --task meat_off_grill # 效果很差很差
python3 RLBench/tools/task_builder_sawyer.py --task setup_chess

# demo generate  
python3 RLBench/tools/dataset_generator_hdf5.py \
    --save_path Datasets \
    --tasks push_button \
    --variations 1 \
    --episodes_per_task 50 \
; \

# visualize episode
python3 RPT_model/visualize_episodes.py --dataset_dir Datasets/sorting_program5/variation0 --episode_idx 3
# modify datasets
python3 RPT_model/mod_datasets.py

# ACT
python RPT_model/imitate_inference.py \
    --task_name push_button \
    --ckpt_dir Trainings \
    --policy_class ACT3E3 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0

# CNNMLP
python RPT_model/imitate_inference.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class CNNMLP --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone resnet18
    
# Diffusion Policy
python RPT_model/imitate_inference.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class Diffusion --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone resnet18


python3 RPT_model/imitate_inference.py \
          --task_name sorting_program5 \
          --ckpt_dir Trainings \
          --policy_class Diffusion --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
          --num_epochs 1000  --lr 1e-5 --seed 0 --backbone resnet18 \
          --eval --temporal_agg 
    
## train and eval
###########################################################################################################################
# task close_jar

python RPT_model/imitate_inference.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_inference.py \
    --task_name push_button \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_inference.py \
    --task_name push_button \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \
    
    