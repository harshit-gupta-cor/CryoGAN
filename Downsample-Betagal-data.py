import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import mrcfile
import scipy
from scipy import ndimage
import torch
from scipy import misc

pathParticle='./Datasets/betagal/Particles/'
pathtoSendParticle='./Datasets/betagal/Particles-384/'
pathBackground='./Datasets/betagal/Background/'
pathtoSendBackground='./Datasets/betagal/Background-384/'

total=41123
for i in range(total):
    print(i)
    pathCurrentParticle=pathParticle+str(i).zfill(6)+".mrc"
    pathCurrenttoSendParticle=pathtoSendParticle+str(i).zfill(6)+".mrc"
    
    im=mrcfile.open(pathCurrentParticle).data
    imd=torch.nn.functional.avg_pool2d(torch.Tensor(im).cuda().unsqueeze(0).unsqueeze(0), kernel_size=2).squeeze().cpu().numpy()
    with mrcfile.new(pathCurrenttoSendParticle, overwrite=True) as m:
        m.set_data(imd)
        
        
    pathCurrentBackground=pathBackground+str(i).zfill(6)+".mrc"
    pathCurrenttoSendBackground=pathtoSendBackground+str(i).zfill(6)+".mrc"
    
    im=mrcfile.open(pathCurrentBackground).data
    imd=torch.nn.functional.avg_pool2d(torch.Tensor(im).cuda().unsqueeze(0).unsqueeze(0), kernel_size=2).squeeze().cpu().numpy()
    with mrcfile.new(pathCurrenttoSendBackground, overwrite=True) as m:
        m.set_data(imd)
    