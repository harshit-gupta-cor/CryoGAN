import os, sys
sys.path.append(os.getcwd())
import torch as th
import torch 
import torchvision as tv
import CryoGAN_Clean as pg
import dataSet_Clean as dataSet
import argparse
from config import Config as cfg
from IPython.core.debugger import set_trace
from copy import deepcopy

def initDataset(args):
       
        data_transform = torch.Tensor

        if args.dataset.startswith('CubePhantom'):
            dataset = dataSet.Phantom(args.dataset)
        elif args.dataset.startswith('TrpV1Phantom'):
            dataset = dataSet.Phantom(args.dataset)           
        elif args.dataset.startswith('spiral'):
            dataset = dataSet.spiral(args.dataset)           
            
        else:
            dataset = dataSet.Cryo(args=args)
        return dataset



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default="Configs/default.cfg", help="Specify config file", metavar="FILE")
    args = parser.parse_args()
    cfg=cfg(args.config)
    #cfg.load_config(args.config)
    #cfg.calc_derived_params()

    # select the device to be used for training
    cfg.device = [th.device('cpu'), th.device('cpu')]
    if th.cuda.is_available():
        if cfg.use2gpu: 
            cfg.device = [th.device('cuda:0'), th.device('cuda:1')]
        else:
            cfg.device = [th.device('cuda'), th.device('cuda')]
    
    # some parameters:
    
    dataset=initDataset(cfg)


    pro_gan = pg.ProGAN(args=cfg, filename=args.config)
  

    pro_gan.train(dataset=dataset, num_workers=1)
