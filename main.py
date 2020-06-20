import os, sys
sys.path.append(os.getcwd())
import torch as th
import torch 
import torchvision as tv
import CryoGAN_Clean as cg
import dataSet_Clean as dataSet
import argparse
from config import Config as cfg
from IPython.core.debugger import set_trace
from copy import deepcopy


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default="Configs/default.cfg", help="Specify config file", metavar="FILE")
    args = parser.parse_args()
    cfg=cfg(args.config)
    
    # select the device to be used for training
    cfg.device = [torch.device('cpu')]
    if torch.cuda.is_available(): cfg.device = [torch.device('cuda')]
    
    # some parameters: 
    dataset = dataSet.Cryo(args=args)
    CryoGAN = cg.CryoGAN(args=cfg, filename=args.config)
    CryoGAN.train(dataset=dataset)
