# -*- coding: utf-8 -*-
"""CycleGAN-GenerativeAI-Icing
Reference for CycleGAN original model (and original pre-trained models used - such as summer2winter): (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). This notebook intends to generate synthetic data for domain transfer of images towards improving the generalisability of AI models for detecting ice in rotor blades of wind turbines.

# Install
"""

!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')

!pip install -r requirements.txt

!pwd

!bash ./datasets/download_cyclegan_dataset.sh summer2winter_yosemite

#This code is to be executed if we get an error when running the training part of the code
!python -m visdom.server

!bash ./scripts/download_cyclegan_model.sh summer2winter_yosemite

#ModifyARotorToIce training
!python train.py --dataroot ModifyARotorToIce/ --load_size 256 --crop_size 256 --n_epochs 100 --n_epochs_decay 100  --batch_size 16 --checkpoints_dir ModifyARotorToIceCheckpoints --name modify_arotor2icemodel --model cycle_gan --use_wandb

!cp ModifyARotorToIceCheckpoints/modify_arotor2icemodel/latest_net_G_A.pth ModifyARotorToIceCheckpoints/modify_arotor2icemodel/latest_net_G.pth

!python test.py --dataroot A-Final/Rotorblade --num_test 920 --checkpoints_dir ModifyARotorToIceCheckpoints/ --results_dir Generated-ModifyARotorToIce-Final --name modify_arotor2icemodel --model test --no_dropout --use_wandb

#ModifyBRotorToIce training
!python train.py --dataroot ModifyBRotorToIce/ --load_size 256 --crop_size 256 --n_epochs 100 --n_epochs_decay 100 --batch_size 16 --checkpoints_dir ModifyBRotorToIceCheckpoints --name modify_brotor2icemodel --model cycle_gan --use_wandb

!cp ModifyBRotorToIceCheckpoints/modify_brotor2icemodel/latest_net_G_A.pth ModifyBRotorToIceCheckpoints/modify_brotor2icemodel/latest_net_G.pth

!python test.py --dataroot B-Final/Rotorblade --num_test 820 --checkpoints_dir ModifyBRotorToIceCheckpoints/ --results_dir Generated-ModifyBRotorToIceFinal --name modify_brotor2icemodel --model test --no_dropout --use_wandb

#Using the pre-trained Summer2Winter model
!python test.py --dataroot ASummerToWinter/testA --checkpoints_dir ASummerToWinterCheckpoints --results_dir ASummerToWinterResults --name summer2winter_spiztner --model test --no_dropout --use_wandb
