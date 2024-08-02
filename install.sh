# 1. conda insatll and create
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh && rm Miniconda3-latest-Linux-x86_64.sh && source ~/.bashrc
    # 1.1recommend setting：
conda config --set auto_activate_base false
conda install -c conda-forge conda-bash-completion
    # 1.2add bash command to bashrc
if [[ -r ~/miniconda3/etc/profile.d/bash_completion.sh ]];then
	. ~/miniconda3/etc/profile.d/bash_completion.sh
else 
	echo "WARNING: could not find conda-bash-completion setup script"
fi

    # 1.3 create env for RLBench_RPT
conda create -n rlbench_rpt python=3.8.10 # the version is strict
conda activate rlbench_rpt


# 2. install CoppeliaSim
    # set env variables
export COPPELIASIM_ROOT=${HOME}/workspace/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
bash $COPPELIASIM_ROOT/coppeliaSim.sh  


# 3. RPT for RLBench 

    # 3.1 git the project
conda activate rlbench_rpt
git clone https://github.com/Boxjod/RLBench_RPT.git
cd RLBench_RPT
    # 3.2 install all requirements
conda activate rlbench_rpt
pip3 install -r requirements.txt
pip3 install -e ./PyRep # need COPPELIASIM_ROOT
pip3 install -e ./RLBench
pip3 install -e ./RPT_model
pip3 install -e ./RPT_model/detr

# 4. test RLBench RPT

    # 4.1 test RLBench task builder
conda activate rlbench_rpt
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5 #[remember don't save scene in Coppeliasim]

    # 4.2 get demo for RPT in RLBench
conda activate rlbench_rpt
python3 RLBench/tools/dataset_generator_hdf5.py \
    --save_path Datasets \
    --tasks sorting_program5 \
    --variations 1 \
    --episodes_per_task 50
    
    python3 RPT_model/visualize_episodes.py --dataset_dir Datasets/sorting_program5/variation0 --episode_idx 0
    
    # 4.3 train task and eval
conda activate rlbench_rpt
python RPT_model/imitate_for_action.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    ; \
python RPT_model/imitate_for_action.py \
    --task_name sorting_program5 \
    --ckpt_dir Trainings \
    --policy_class ACT3E2 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 1000  --lr 1e-5 --seed 0 --backbone efficientnet_b0 \
    --eval --temporal_agg \
    ; \


# 5. recommend tools
HDF5_Viewer: https://myhdf5.hdfgroup.org/
