# 1. conda insatll
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh && rm Miniconda3-latest-Linux-x86_64.sh && source ~/.bashrc

conda create -n rlbench_rpt python=3.8.10 # the version is strict
conda activtae rlbench_rpt

# 2. CoppeliaSim
# set env variables
export COPPELIASIM_ROOT=${HOME}/workspace/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

bash /home/ubuntu/data0/liangws/peract_root/CoppeliaSim/coppeliaSim.sh  

# 3. RPT for RLBench 
# 3.1 git the project
git clone https://github.com/Boxjod/RLBench_RPT.git
# 3.2 install and test pyrep
cd RLBench_RPT/PyRep
pip3 install -r requirements.txt
pip3 install -e .
python examples/example_panda_reach_target.py
# 3.3 install andtest RLBench
cd RLBench_RPT/RLBench
pip install -r requirements.txt
pip install -e .
python tools/task_builder.py
# 3.4 test RLBench RPT
cd RLBench_RPT/RPT_Model
pip install -r requirements.txt
cd act/detr && pip install -e .
python3 RLBench/tools/dataset_generator_sawyer_act3.py \
    --save_path Datasets \
    --tasks sorting_program \
    --variations 1 \
    --episodes_per_task 50 

# 4. tools
HDF5_Viewer: https://myhdf5.hdfgroup.org/
