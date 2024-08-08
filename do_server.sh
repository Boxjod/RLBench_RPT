#!/bin/bash

# tmux new -s xj
#  ctrl+B "：" set -g mouse on 
# conda activate xj_rlbench

# tmux a -t xj
# . do_server.sh

git checkout .
git pull

bash train_eval.sh