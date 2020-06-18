from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch import Tensor


from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer
import torch.nn.init as init
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import Tensor
import astra
import numpy as np



def SymmetryFunction(xd,N=64,d=7, mode='d'):
   # N=N+d-N%d

    if mode=='d':
        Nbig=xd.size()[2]

        xd_a=xd.narrow(1,0,xd.size()[1]//2)
        xd_b=xd.narrow(1,xd.size()[1]//2,xd.size()[1]//2)
        xd_aflip=torch.flip(xd_a,[1])
        xd_bflip=torch.flip(xd_b,[1])
        xd_afabbf=torch.cat([xd_aflip,xd_a,xd_b,xd_bflip],1)
        xd_bbfafa=torch.flip(torch.cat([xd_b,xd_bflip,xd_aflip,xd_a],1),[0])

        xdFull=torch.cat([xd_afabbf,xd_bbfafa],0)



        xpol=(xdFull.unsqueeze(1)+torch.zeros(xdFull.size()[0],d,xdFull.size()[1],xdFull.size()[2]).cuda()).view(xdFull.size()[0],-1,xdFull.size()[2])

        nx=Nbig-N

        return PolToCart(xpol,n=Nbig).narrow(1,nx//2,N).narrow(2,nx//2,N)
    elif mode=='c':
        
        Nbig=xd.size()[2]
        
       
        xd_flip=torch.flip(xd,[0])
        
        xdFull=torch.cat([xd,xd_flip],0)
        
        xpol=(xdFull.unsqueeze(1)+torch.zeros(xdFull.size()[0],d,xdFull.size()[1],xdFull.size()[2]).cuda()).view(xdFull.size()[0],-1,xdFull.size()[2])

        nx=Nbig-N

        return PolToCart(xpol,n=Nbig).narrow(1,nx//2,N).narrow(2,nx//2,N)
        
def SymCreatorD2(xd1):
   

    xd2=torch.flip(xd1, [2,3])
 
    im1=torch.cat([xd1, xd2],2)
    im2=torch.flip(im1,[1,2])
    im=torch.cat([im1,im2], 1)
    return im

def UnSymCreatorD2(x):
   
    coords=x.shape
    
    return x[:,:coords[1]//2,:coords[2]//2, :]
        
def SymCreatorC4(xd1):
   
   
    xd2=torch.flip(xd1, [1])
    xd2=torch.transpose(xd2, 2,1)

    xd3=torch.flip(torch.flip(xd1,[1]),[2])
    
    xd4=torch.flip(xd1, [2])
    xd4=torch.transpose(xd4, 2,1)
    
    im1=torch.cat([xd1, xd2],2)
    im2=torch.cat([xd4, xd3],2)
    im=torch.cat([im1,im2], 1)
   
    return im

def SymCreatorC2(xd1):
    
    xd2=torch.flip(torch.flip(xd1,[1]),[2])    
    im=torch.cat([xd1,xd2], 1)
    return im


def UnSymCreatorC4(x):
   
    coords=x.shape
    
    return x[:,:coords[1]//2,:coords[2]//2, :]


def UnSymCreatorC2(x):
   
    coords=x.shape
    
    return x[:,:coords[1]//2,:, :]
   # im_flip=torch.flip(im,[0])
   # return torch.cat([im_flip,im],0)
 

def CartToPolGrid(n,device):
    tt,rr=torch.meshgrid(torch.arange(0,n).to(device),torch.arange(0,n).to(device))
   
    rr=rr.float().contiguous().view(-1,1)/np.sqrt(2)
    tt=tt.float().contiguous().view(-1,1)*np.pi*2/(n-1)
    
    xx=0.5*n-rr*torch.sin(tt)#rr.float()/(n *torch.cos(tt.float()*2*np.pi/n)
    yy=0.5*n+rr*torch.cos(tt)#rr.float()/n *torch.sin(tt.float()*2*np.pi/n)
    
    grid=torch.cat([-1+2.0*xx/n,-1+2.0*yy/n],1).view(n,n,2)
    return grid

def PolToCartGrid(n, device):
    xx,yy=torch.meshgrid(torch.arange(0,n).to(device),torch.arange(0,n).to(device))
  
    
    yy=yy.float().contiguous().view(-1,1)-(n//2 - 1)
    xx=xx.float().contiguous().view(-1,1)-(n//2 - 1)
    
    rr=torch.sqrt(2*(xx.pow(2)+yy.pow(2)))#-1+ (2**0.5)*2*((xx.float()/n -0.5).pow(2)+(yy.float()/n -0.5).pow(2)).sqrt()
    tt=(torch.atan2(yy,xx)+np.pi)*n/(2*np.pi)#-1+2*((yy>=n//2).float()*np.pi/2.0+torch.atan((xx.float()-n//2).float()/(yy.float()-n//2 + 1e-12).float())/2.0+np.pi/4.0)/np.pi 
    
    grid=torch.cat([-1 + 2.0*rr/n,-1 + 2.0*tt/n],1).view(n,n,2)
    return grid

def CartToPol(x,n,grid=None):
    if grid is None:
        grid=CartToPolGrid(n,x.device) +  torch.zeros(x.shape[0],1,1,2).to(x.device)
    
    xpol= F.grid_sample(x, grid, mode='bilinear', padding_mode='border')
    return xpol

def PolToCart(xpol,n,grid=None):
    if grid is None:
        grid=PolToCartGrid(n,xpol.device)+  torch.zeros(xpol.shape[0],1,1,2).to(xpol.device)
    
    x= F.grid_sample(xpol, grid, mode='bilinear', padding_mode='border')
    return x