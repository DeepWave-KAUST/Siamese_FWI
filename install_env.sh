#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh
# 

echo 'Creating Package environment'

# create conda env
conda env create -f environment.yml
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh

conda activate my_env
echo 'Created and activated environment:' $(which python)

# Check torch OR tensorflow works as expected
echo 'Checking torch version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'
# TODO convert below line to work for tensorflow test
# echo 'Checking tensorflow version and running a command...'
# python -c 'import tensorflow; print(tensorflow.__version__);  print(tensorflow.cuda.get_device_name(tensorflow.cuda.current_device())); print(tensorflow.ones(10).to("cuda:0"))'

echo 'Done!'

