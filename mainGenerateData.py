import os, sys
sys.path.append(os.getcwd())
import torch as th
import torch 
import torchvision as tv
import GenerateData as gd
import argparse
from config import Config as cfg



# select the device to be used for training
device = th.device("cuda" if th.cuda.is_available() else "cpu")




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default="Configs/default.cfg", help="Specify config file", metavar="FILE")
    args = parser.parse_args()
    cfg=cfg(args.config)
    



    # some parameters:
    depth = 1
    # hyper-parameters per depth (resolution)
    res=        [256]
    num_epochs = [cfg.DatasetSize]
    fade_ins = [ 0]
    batch_sizes = [ 1]
    latent_size = 64

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    gd = gd.GenerateData(device=device, args=cfg)
    # ======================================================================

    # ======================================================================
    # This line trains the PRO-GAN
    # ======================================================================
    gd.generate()