# 1. conda insatll
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh && rm Miniconda3-latest-Linux-x86_64.sh && source ~/.bashrc

conda create -n rlbench_rpt python=3.8.10 # the version is strict
conda activtae rlbench_rpt

# 2. CoppeliaSim
# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

bash /home/ubuntu/data0/liangws/peract_root/CoppeliaSim/coppeliaSim.sh  

# 3. PyRep
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip3 install -r requirements.txt
pip3 install -e .
python examples/example_panda_reach_target.py

# 4. RLBench
git clone https://github.com/stepjam/RLBench.git
cd RLBench
pip install -r requirements.txt
pip install -e .
python tools/task_builder.py

# 5. ACT
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython

git clone https://github.com/tonyzhaozh/act.git
cd act/detr && pip install -e .
CUDA_VISIBLE_DEVICES=7 python3 record_sim_episodes.py \
--task_name sim_transfer_cube_scripted \
--dataset_dir data/sim_transfer_cube_scripted \
--num_episodes 1
python3 visualize_episodes.py --dataset_dir data/sim_transfer_cube_scripted --episode_idx 0

# 6. RLBench_ACT_Sawyer
git clone https://github.com/Boxjod/RLBench_ACT_Sawyer.git

# 7. tools
HDF5_Viewer: https://myhdf5.hdfgroup.org/

# 8. LLMs
pip install transformers #没问题
pip install h5py_cache

conda install transformers #有问题了
pip uninstall charset-normalizer
pip install charset-normalizer
pip install chardet

###########################################################################################################
conda install transformers
Retrieving notices: ...working... done
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/boxjod/miniconda3/envs/xj_rlbench

  added / updated specs:
    - transformers


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    abseil-cpp-20211102.0      |       hd4dd3e8_0        1020 KB  defaults
    aiohttp-3.9.3              |   py38h5eee18b_0         707 KB  defaults
    aiosignal-1.2.0            |     pyhd3eb1b0_0          12 KB  defaults
    arrow-cpp-11.0.0           |       hda39474_2        10.2 MB  defaults
    async-timeout-4.0.3        |   py38h06a4308_0          12 KB  defaults
    attrs-23.1.0               |   py38h06a4308_0         140 KB  defaults
    aws-c-common-0.4.57        |       he6710b0_1         156 KB  defaults
    aws-c-event-stream-0.1.6   |       h2531618_5          25 KB  defaults
    aws-checksums-0.1.9        |       he6710b0_0          49 KB  defaults
    aws-sdk-cpp-1.8.185        |       hce553d0_0         1.9 MB  defaults
    blas-1.0                   |         openblas          46 KB  defaults
    boost-cpp-1.82.0           |       hdb19cb5_2          11 KB  defaults
    bottleneck-1.3.7           |   py38ha9d4c09_0         125 KB  defaults
    datasets-2.12.0            |   py38h06a4308_0         636 KB  defaults
    dill-0.3.6                 |   py38h06a4308_0         167 KB  defaults
    frozenlist-1.4.0           |   py38h5eee18b_0          52 KB  defaults
    gflags-2.2.2               |       h6a678d5_1         145 KB  defaults
    glog-0.5.0                 |       h6a678d5_1         107 KB  defaults
    grpc-cpp-1.48.2            |       h5bf31a4_0         4.8 MB  defaults
    huggingface_hub-0.20.3     |   py38h06a4308_0         392 KB  defaults
    icu-73.1                   |       h6a678d5_0        25.9 MB  defaults
    importlib-metadata-7.0.1   |   py38h06a4308_0          40 KB  defaults
    libboost-1.82.0            |       h109eef0_2        19.5 MB  defaults
    libbrotlicommon-1.0.9      |       h5eee18b_7          70 KB  defaults
    libbrotlidec-1.0.9         |       h5eee18b_7          31 KB  defaults
    libbrotlienc-1.0.9         |       h5eee18b_7         264 KB  defaults
    libevent-2.1.12            |       h8f2d780_0         425 KB  defaults
    libgfortran-ng-11.2.0      |       h00389a5_1          20 KB  defaults
    libgfortran5-11.2.0        |       h1234567_1         2.0 MB  defaults
    libopenblas-0.3.21         |       h043d6bf_0         5.4 MB  defaults
    libthrift-0.15.0           |       h0d84882_2         4.0 MB  defaults
    multidict-6.0.4            |   py38h5eee18b_0          54 KB  defaults
    multiprocess-0.70.14       |   py38h06a4308_0         238 KB  defaults
    numexpr-2.8.4              |   py38hd2a5715_1         136 KB  defaults
    numpy-1.24.3               |   py38hf838250_0          11 KB  defaults
    numpy-base-1.24.3          |   py38h1e6e340_0         6.9 MB  defaults
    orc-1.7.4                  |       hb3bc3d3_1         972 KB  defaults
    packaging-23.2             |   py38h06a4308_0         145 KB  defaults
    pandas-2.0.3               |   py38h1128e8f_0        12.4 MB  defaults
    pyarrow-11.0.0             |   py38h468efa6_1         4.3 MB  defaults
    python-dateutil-2.8.2      |     pyhd3eb1b0_0         233 KB  defaults
    python-tzdata-2023.3       |     pyhd3eb1b0_0         140 KB  defaults
    python-xxhash-2.0.2        |   py38h5eee18b_1          21 KB  defaults
    pytz-2023.3.post1          |   py38h06a4308_0         209 KB  defaults
    re2-2022.04.01             |       h295c915_0         210 KB  defaults
    regex-2023.10.3            |   py38h5eee18b_0         367 KB  defaults
    responses-0.13.3           |     pyhd3eb1b0_0          24 KB  defaults
    safetensors-0.4.2          |   py38ha89cbab_0         1.1 MB  defaults
    six-1.16.0                 |     pyhd3eb1b0_1          18 KB  defaults
    snappy-1.1.10              |       h6a678d5_1          43 KB  defaults
    tokenizers-0.13.2          |   py38he7d60b5_1         4.1 MB  defaults
    tqdm-4.65.0                |   py38hb070fc8_0         131 KB  defaults
    transformers-4.32.1        |   py38h06a4308_0         6.7 MB  defaults
    typing-extensions-4.9.0    |   py38h06a4308_1           9 KB  defaults
    utf8proc-2.6.1             |       h5eee18b_1          93 KB  defaults
    xxhash-0.8.0               |       h7f8727e_3          83 KB  defaults
    yarl-1.9.3                 |   py38h5eee18b_0         117 KB  defaults
    zipp-3.17.0                |   py38h06a4308_0          21 KB  defaults
    ------------------------------------------------------------
                                           Total:       117.0 MB
