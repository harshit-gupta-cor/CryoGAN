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

from .FunctionsSaveImage import *

def fftshift(x, mode='complex', signal_dim=2):# Shift  imagesSize//2 to the top left corner
    dims=[len(x.shape)-i -1 -1 * (mode=='complex') for i in range(signal_dim) ]
    dims=dims[::-1]
    shifts=[x.shape[dim]//2 + x.shape[dim]%2 for dim in dims]
    out=torch.roll(x, shifts=shifts, dims=dims)
    return out # last dim=2 (real&imag)   


def ifftshift(x, mode='complex', signal_dim=2):# Shift  imagesSize//2 to the top left corner # mode complex or real
    dims=[len(x.shape)-i -1 -1 * (mode=='complex') for i in range(signal_dim) ]
    shifts=[x.shape[dim]//2 for dim in dims]
    out=torch.roll(x, shifts=shifts, dims=dims)
    return out # last dim=2 (real&imag)        
    
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):# Shift the top left corner to imagesSize//2
    # assumes that dim one is the batch and does not shift it!
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):# Shift  imagesSize//2 to the top left corner
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1) # last dim=2 (real&imag)        

def batch_fftshift2dreal(x):# Shift the top left corner to imagesSize//2
    # assumes that dim one is the batch and does not shift it!
    real=x
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        
    return  real 

def batch_ifftshift2dreal(x):# Shift  imagesSize//2 to the top left corner
    real=x
    for dim in range(len(real.size()) , 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        
    return real     

def SpaceToFourier(X, signal_dim=3, normalized=True):
    XF=fftshift(torch.rfft(ifftshift(X, mode='real', signal_dim=signal_dim),signal_dim, onesided=False, normalized=normalized), mode='complex', signal_dim=signal_dim)
    return XF

def FourierToSpace(XF, signal_dim=3, normalized=True):
    X=fftshift(torch.irfft(ifftshift(XF, mode='complex', signal_dim=signal_dim),signal_dim, onesided=False, normalized=normalized), mode='real', signal_dim=signal_dim)
    return X

#%%

def FFTConv(imgs, filt, plot=False):
    # take image of size B*C*H*W and filts of size 1*1*H*W, both real
    # filter should have odd dimensions
    # center is assumed at ceil(size/2)
    #
    # output: size B*C*H*W
    
    filtSize = np.array(filt.size()[2:])
    if np.any(filtSize % 2 == 0):
        raise TypeError("filter size {} should be odd".format(filtSize))

    imSize = np.array(imgs.size()[2:])


    # zero pad
    # Pad arg = (last dim pad left side, last dim pad right side, 2nd last dim left side, etc..)

    fftSize = imSize + filtSize - 1
    
    imgs  = F.pad(imgs, (0, fftSize[0]-imSize[0],   0, fftSize[1]-imSize[1]))
    filt  = F.pad(filt, (0, fftSize[0]-filtSize[0], 0, fftSize[1]-filtSize[1]))

    # shift the center to the upper left corner
    filt = roll_n(filt, 2, filtSize[0]//2)
    filt = roll_n(filt, 3, filtSize[1]//2)

    
    imgsFourier = torch.rfft(imgs,2, onesided=False) # rfft doesn't require complex input
    filtFourier = torch.rfft(filt, 2, onesided=False)

    # Extract the real and imaginary parts
    imgR,imgIm = torch.unbind(imgsFourier , -1)
    filtR,filtIm = torch.unbind(filtFourier , -1)

    if plot==True:
    
        save_fig_double(filtR.data.cpu(),filtIm.data.cpu(), './', 'CurrentCTF-Fourier',  iteration=None, Title1='Real', Title2='Imag' )
        save_fig_double((imgR+1e-8).abs().log().data.cpu(),(imgIm+1e-8).abs().log().data.cpu(), './', 'CurrentProj-Fourier', iteration=None, Title1='Real', Title2='Imag' )
        
    # Do element wise complex multiplication
    imgFilterdR=imgR*filtR-imgIm*filtIm
    imgFilteredIm=imgIm*filtR+imgR*filtIm

    imgFiltered = torch.stack((imgFilterdR, imgFilteredIm), -1)
    imgFiltered = torch.irfft(imgFiltered,2, onesided=False)



    return imgFiltered[:,:, :imSize[0], :imSize[1]], imgsFourier, filtFourier,filt


def maskShell(grid, r, w=1):
    mask=(grid-r).abs()< w
    return mask.float()


def RadiallyAverage(x, dim=2, w=0.5):
    
    n=x.shape[-1]//2
    vec=torch.arange(-n,n+x.shape[-1]%2).to(x.device)
    if dim==2:
        xx, yy= torch.meshgrid(vec, vec)
    else:
        xx, yy, zz= torch.meshgrid(vec, vec, vec)

    xx=xx.contiguous().view(-1).float()
    yy=yy.contiguous().view(-1).float()
    
    grid2=xx.pow(2)+yy.pow(2)
    if dim !=2:
        zz=zz.contiguous().view(-1).float()
        grid2=grid2+zz.pow(2)
        
    grid=grid2.sqrt()
    if dim !=2:
        grid=grid.view(x.shape[-3], x.shape[-2], x.shape[-1])
    else:
        grid=grid.view( x.shape[-2], x.shape[-1])
        
        
    fsc=torch.zeros( x.shape[0], n)
    for r in range(n):
        shell=maskShell(grid, r)

        fsc[:,r]=(shell*x).abs().view(x.shape[0], -1).sum(1)/(shell.sum()+1e-12)
    return fsc

def RadiallyAverageFourierTransform(x, dim=2):
    real, imag= torch.unbind(SpaceToFourier(x, signal_dim=dim), dim=-1)
    xF=(real.pow(2)+imag.pow(2)).sqrt()
    fsc=RadiallyAverage(xF, dim)
    return fsc


def softmask(size, radius, width,device, signal_dim=3, mode='lp' ):
    n=size
    vec=torch.arange(-n//2,n//2).to(device)
    
    if signal_dim==3:
       xx,yy,zz= torch.meshgrid(vec, vec, vec)
       disc=(xx.pow(2)+yy.pow(2)+zz.pow(2)).float().sqrt()
    elif signal_dim==2:
       xx,yy= torch.meshgrid(vec, vec)
       disc=(xx.pow(2)+yy.pow(2)).float().sqrt()
    
    radius=n/2.0*radius
    width=n/2.0*width
    radiuslow=radius-width/2.0
    radiushigh=radius+width/2.0
    
    maskcos= (disc<=radiushigh).float()*(radiuslow<=disc).float()*torch.cos(np.pi*(disc-radiuslow)/width)
    mask=((disc<=radiushigh).float()*(1+maskcos)+(disc<radiuslow).float())/2.0
    
    if mode=='lp':
        mask=mask
    elif mode=='bp':
        mask=mask *(1-mask)
        mask=mask/mask.max()
    elif mode=='hp':
        mask=(1-mask)
    return mask,disc.float()/(n/2)

def SoftMaskFiltering(im, radius, width, signal_dim, mode):
    imF=SpaceToFourier(im, signal_dim=signal_dim)
    mask,disc=softmask(im.shape[-1], radius=radius, width=width,device=im.device, signal_dim=signal_dim )
    maskstacked=torch.stack([mask, mask],-1)
    filtered= imF*maskstacked
    return FourierToSpace(filtered, signal_dim=signal_dim), filtered, mask, disc


