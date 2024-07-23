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
conda activtae rlbench_rpt


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
git clone https://github.com/Boxjod/RLBench_RPT.git
cd RLBench_RPT
    # 3.2 install all requirements
conda activate rlbench_rpt
pip3 install -r requirements.txt
pip3 install -e ./PyRep # need COPPELIASIM_ROOT
pip3 install -e ./RLBench
pip install -e ./RPT_Model
pip install -e ./RPT_Model/detr

    # 3.3 test RLBench RPT
python3 RLBench/tools/task_builder_sawyer.py --task sorting_program5

python3 RLBench/tools/dataset_generator_sawyer_act3.py \
    --save_path Datasets \
    --tasks sorting_program5 \
    --variations 1 \
    --episodes_per_task 50


cd RLBench_RPT/RPT_Model
cd act/detr && pip install -e .



python3 RLBench/tools/dataset_generator_sawyer_act3.py \
    --save_path Datasets \
    --tasks sorting_program \
    --variations 1 \
    --episodes_per_task 50 

# 4. tools
HDF5_Viewer: https://myhdf5.hdfgroup.org/
