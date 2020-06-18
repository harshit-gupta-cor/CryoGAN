#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:18:31 2019

@author: Harshit
"""

from __future__ import print_function
import argparse
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

from .Functions import *
from .FunctionsCTF import *
from .FunctionsSymmetry import *
from .FunctionsFourier import *




import scipy.special
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



from torch.utils.data.dataset import Dataset
from  torch.utils.data.dataloader import DataLoader
import mrcfile
#%%

    
def CTFforward(h, x):

    paddingLength=h.size()[-1]//2
    
    pad=(paddingLength,paddingLength, paddingLength, paddingLength)
    
    xpadded=F.pad(x, pad, "constant", 0)
    
    xConvolved=FFTConv(xpadded, h, plot=doPlots) [0][:,:,paddingLength: paddingLength+x.size()[-2],  paddingLength:paddingLength+x.size()[-1]]

    return xConvolved
    


def Projection(X, RawProjectionSize, vectors ):
    
    proj = Project.apply(X, RawProjectionSize, vectors)
    return proj


                    
def InitArgs(args):

     #init args
  
    args.ConvMode=False


    # init ctf, vol geom and X
    if args.CTF:
        if args.UseEstimatedAngles or args.UseEstimatedDefocuses or args.UseEstimatedTranslations:
             args.imageNum=np.random.randint(0,args.EstimatedDataSize,args.BATCH_SIZE) 

        args.h,args.hFourier= CTFGeneratorSpatial(args)     

    args.vol_geom=astra.create_vol_geom(args.VolumeSize,args.VolumeSize, args.VolumeSize)

    args.dc=Tensor([args.dc]).cuda()
    args.sigma1=Tensor([args.sigma1]).cuda()
    args.sigma2=Tensor([args.sigma2]).cuda()
    args.scalar=Tensor([args.scalar]).cuda()
    
    args.dc=nn.Parameter(args.dc.clone().cuda())
    args.sigma1=nn.Parameter(args.sigma1.clone().cuda())
    args.sigma2=nn.Parameter(args.sigma2.clone().cuda())
    args.scalar=nn.Parameter(args.scalar.clone().cuda())

    args.U2T=lambda x: (x-0.5).sign()*(1-(1-(2*x-1).abs()).sqrt())
    astra.astra.set_gpu_index([ torch.Tensor([1]).cuda().device.index])
    return args


   
        
def InitVolume( args):  
    X=None
    Xd=None
    S=None
    UnS=None
    if args.SymmetryType=='none':
        X=Tensor(args.VolumeNumbers, args.VolumeSize, args.VolumeSize, args.VolumeSize).fill_(0).cuda()
        if args.VolumeDomain =='fourier':
            X=SpaceToFourier(X, signal_dim=3)
        if args.UseVolumeGenerator == False:
            X=nn.Parameter(X)

    else:
        n=args.SymmetryN
        if args.SymmetryType == 'D' and n==2:
            S= lambda x: SymCreatorD2(x)
            UnS= lambda x: UnSymCreatorD2(x)
            Xd=Tensor(args.VolumeNumbers,(args.VolumeSize)//2,(args.VolumeSize)//2,(args.VolumeSize)).fill_(0).cuda()
            print("using D2 symetry!!")
        elif args.SymmetryType =='D':
            ####TODO: put symmetric function in the self
            S=lambda x:SymmetryFunction(x, N=args.VolumeSize,d=n, mode='d')
            UnS= lambda x: UnSymmetryFunction(x, N=args.VolumeSize,d=n, mode='d')
            Nbig=(args.VolumeSize)+4*n-(args.VolumeSize)%(4*n)
            Xd=Tensor(args.VolumeNumbers,(args.VolumeSize)//2,Nbig//(2*n),Nbig).fill_(0).cuda()

        elif args.SymmetryType=='C' and n==4:
            
            S=lambda x:SymCreatorC4(x)
            UnS=lambda x:UnSymCreatorC4(x)
            Nbig=(args.VolumeSize)
            Xd=Tensor(args.VolumeNumbers,Nbig//2,Nbig//2, args.VolumeSize).fill_(0).cuda()
            
        elif args.SymmetryType=='C' and n==2:
            
            S=lambda x:SymCreatorC2(x)
            UnS=lambda x:UnSymCreatorC2(x)
            
            Nbig=(args.VolumeSize)
            Xd=Tensor(args.VolumeNumbers,Nbig//2,Nbig, args.VolumeSize).fill_(0).cuda()

        else:
            S=Symmetry_Func.apply
            Xd=Tensor(args.VolumeNumbers,(args.VolumeSize//args.d),(args.VolumeSize//args.d),2*(args.VolumeSize//args.d)).fill_(0).cuda()
        
        if args.VolumeDomain =='fourier':
            Xd=SpaceToFourier(Xd, signal_dim=3)
            
            
        if args.UseVolumeGenerator == False:
            Xd=nn.Parameter(Xd)
        
    return X, Xd, S, UnS

            
def InitMask( args, Xd=None, X=None, radius=None):

    mask=None
    maskd=None
    if args.SymmetryType=="D":
      #  raise Exception('This case is unlikely to work correctly, please check it')
       

        maskd=torch.zeros(Xd.shape[-3:]).cuda()
       # maskd[int((1- args.VolumeMaskSize)*Xd.size(0)):,int((1- args.VolumeMaskSize)*Xd.size(0)):,:]=1
        
        
       # maskd=torch.zeros(Xd.size()).cuda()
        xx,yy,zz=torch.meshgrid(torch.arange(0,Xd.size()[1]).cuda(),torch.arange(0,Xd.size()[2]).cuda(),torch.arange(0,Xd.size()[3]).cuda())

        xx=xx.contiguous().view(-1)
        yy=yy.contiguous().view(-1)
        zz=zz.contiguous().view(-1)
        
        centrex=maskd.shape[0]
        centrey=maskd.shape[1]
        centrez=maskd.shape[2]//2
        
        xx2=(xx-centrex).pow(2).float()
        yy2=(yy-centrey).pow(2).float()
        zz2=(zz-centrez).pow(2).float()
        
        ratioxx=1.0
        ratioyy=1.0
        ratiozz=1.0#(1.4*(1/args.VolumeMaskSize))**2
       
        radius=(args.VolumeSize//2)*radius

        valueMask=(xx2/ratioxx+ yy2/ratioyy+zz2/ratioyy).sqrt() <radius
        
        maskd=valueMask.float().cuda().view(Xd.shape[1],Xd.shape[2] , Xd.shape[3])
       

    elif args.SymmetryType=="C" and  args.SymmetryN==4:
        maskd=torch.zeros(Xd.shape[-3:]).cuda()
        xx,yy,zz=torch.meshgrid(torch.arange(0,Xd.size()[1]),torch.arange(0,Xd.size()[2]),torch.arange(0,Xd.size()[3]))

        xx=xx.contiguous().view(-1)
        yy=yy.contiguous().view(-1)
        zz=zz.contiguous().view(-1)
        valueMask=( ((xx-maskd.shape[0]).pow(2)+(yy-maskd.shape[1]).pow(2)).float().sqrt() < np.max([args.VolumeMaskSize, 0.5])*maskd.shape[0])*((zz-maskd.shape[2]//2).float().abs()<      args.VolumeMaskSize  * maskd.shape[2]//2)

        radius=(args.VolumeSize//2)*radius
        valueMask=( ((xx-maskd.shape[0]).pow(2)+(yy-maskd.shape[1]).pow(2)+(zz-maskd.shape[2]//2).pow(2)).float().sqrt() < radius)



        maskd=valueMask.float().cuda().view(Xd.shape[1],Xd.shape[2] , Xd.shape[3])
    
    
    elif args.SymmetryType=="C" and  args.SymmetryN==2:
        maskd=torch.zeros(Xd.shape[-3:]).cuda()
        xx,yy,zz=torch.meshgrid(torch.arange(0,Xd.size()[1]),torch.arange(0,Xd.size()[2]),torch.arange(0,Xd.size()[3]))

        xx=xx.contiguous().view(-1)
        yy=yy.contiguous().view(-1)
        zz=zz.contiguous().view(-1)
        valueMask=( ((xx-maskd.shape[0]).pow(2)+(yy-maskd.shape[1]//2).pow(2)).float().sqrt() < np.max([args.VolumeMaskSize, 0.5])*maskd.shape[0])*((zz-maskd.shape[2]//2).float().abs()<      args.VolumeMaskSize  * maskd.shape[2]//2)

        radius=(args.VolumeSize//2)*radius
        valueMask=( ((xx-maskd.shape[0]).pow(2)+(yy-maskd.shape[1]//2).pow(2)+(zz-maskd.shape[2]//2).pow(2)).float().sqrt() < radius)



        maskd=valueMask.float().cuda().view(Xd.shape[1],Xd.shape[2] , Xd.shape[3])
    
    else:
        mask=Tensor(X.shape[-3:]).cuda().fill_(0)        
        xx,yy,zz=torch.meshgrid(torch.arange(0,X.size()[1]),torch.arange(0,X.size()[2]),torch.arange(0,X.size()[3]))

        xx=xx.contiguous().view(-1)
        yy=yy.contiguous().view(-1)
        zz=zz.contiguous().view(-1)
        centrex=(args.VolumeSize//2)
        centrey=(args.VolumeSize//2)
        centrez=(args.VolumeSize//2)
        radius=(args.VolumeSize//2)*radius
        valueMask=((xx-centrex).pow(2) + (yy-centrey).pow(2) + (zz-centrez).pow(2) ).float().sqrt()<(radius)
        mask=valueMask.float().cuda().view(X.shape[1],X.shape[2] , X.shape[3])
   
    if args.VolumeDomain=='fourier':
        if mask is not None:
            mask=torch.stack((mask,mask), -1)
        else:
            maskd=torch.stack((maskd,maskd), -1)
    
        
    return mask,maskd, xx, yy, zz, centrex, centrey, centrez, radius
   


    
    
def TranslationPadding( x, translationx, translationy):

    paddingLength=torch.max(torch.cat([translationx.abs(), translationy.abs()],-1))        
    pad=(paddingLength,paddingLength, paddingLength, paddingLength)

    xpadForTranslation=F.pad(x, pad, "constant", 0)
    return xpadForTranslation
        
def TranslationCropping( x,translationx, translationy, args )  :

    paddingLength=torch.max(torch.cat([translationx.abs(), translationy.abs()],-1))        
    nx=paddingLength-translationx
    ny=paddingLength-translationy

    with torch.no_grad():
              mask=torch.Tensor(x.size()).fill_(0).byte().cuda()

              for index,maskInd in enumerate(mask,0):
                  maskInd[ 0,  nx[index]  :   args.ProjectionSize+nx[index],   ny[index]  :   args.ProjectionSize+ny[index]]=1

    xCropping=torch.masked_select(x,mask.bool())

    xCropping=xCropping.view(x.size()[0],1,args.ProjectionSize, args.ProjectionSize) 
    
    return xCropping


        
        
class Symmetry_Func(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
  
    
           
    def forward(ctx,Xd):
        
        return  Support.SymCreator(Xd)
       
    
    @staticmethod
    def backward(ctx,grad_out):
        grad_x=None
        
        if ctx.needs_input_grad[0]:
             
           grad_x=Support.SymUncreator(grad_out)
        
       
        return  grad_x


    


class Project(torch.autograd.Function):
    """
    X-ray projections using astra-toolbox
    """

    @staticmethod
    def forward(ctx, input, projSize, vectors):#angles=T(B,3)
        """
        input - GPU tensor
        ctx.vectors - numProj x 12 following astra convention
        ctx.projSize - size in pixels of projections 
        
        returns:
        proj: GPU tensor numProj x 1 x projSize x projSize

        avoid passing back to the cpu
        based on https://github.com/astra-toolbox/astra-toolbox/blob/master/samples/python/s021_pygpu.py
        and https://github.com/astra-toolbox/astra-toolbox/blob/10d87f45bc9311c0408e4cacdec587eff8bc37f8/python/astra/creators.py create_sino3d_gpu
        """

        # pytorch is row major (later dimensions change faster), while astra is col major (opposite)
        # to counteract this, we need to transpose
        # the contiguous is needed to change the memory layout 
      
        astra.set_gpu_index([input.device.index], memory=1*1024*1024*1024)
        
       
        input = input.permute(2, 1, 0).contiguous()

        # setup volume
        x,y,z = input.shape
        
        strideBytes = input.stride(-2) * 32/8 # 32 bits in a float, 8 bits in a byte
        vol_link = astra.data3d.GPULink(input.data_ptr(), x, y, z, strideBytes)
        vol_geom = astra.create_vol_geom(x,y,z)
        vol_id = astra.data3d.link('-vol', vol_geom, vol_link)

        # setup projection
        proj = torch.empty(projSize, len(vectors), projSize,
                           dtype=torch.float, device='cuda')
        x,y,z = proj.shape
        strideBytes = proj.stride(-2) * 32/8 # 32 bits in a float, 8 bits in a byte
        proj_link = astra.data3d.GPULink(proj.data_ptr(), x, y, z, strideBytes)
        proj_geom = astra.create_proj_geom('parallel3d_vec', projSize, projSize, vectors.numpy())
        proj_id = astra.data3d.link('-sino', proj_geom, proj_link)


        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # again handle the row vs col major problem
        # the output is y_0 x numProj x y_1, which pytorch interprets in the reverse order
        proj = proj.permute( 1, 2, 0).unsqueeze(1)

        # store data for back propagation
        ctx.proj_geom = proj_geom
        ctx.vol_geom = vol_geom
        
        return proj
    

    @staticmethod
    def backward(ctx, grad_out):
        astra.set_gpu_index([grad_out.device.index], memory=10*1024*1024*1024)
        
        grad_input = gradProjSize = gradVectors = None
        
        if ctx.needs_input_grad[0]:
            # permute here to handle opposite data layouts
            bproj_id, bproj_data = astra.create_backprojection3d_gpu( grad_out.squeeze(1).permute(2, 0, 1).data.cpu().numpy(), ctx.proj_geom, ctx.vol_geom)
            astra.data3d.delete(bproj_id)  
            grad_input = Tensor(bproj_data).cuda(non_blocking=True).permute( 2, 1, 0 )

        return grad_input, gradProjSize, gradVectors
    


def angles_to_vectors(angles, step=1):
    """
    convert angles in the cryoem convention
    described in "Common conventions for interchange and archiving of three-dimensional electron microscopy information in structural biology"
    to the astra convection vectors
    
    you can think of the object rotating
    first, phi around the z-axis
    second, theta around the global y-axis
    finally, psi around the global z-axis, which just rotates the projection

    projection happens along z-axis
    
    angles: numProj x 3 on CPU: phi, theta, psi in radians!


    returns: vectors for astra

    """

    pi=np.pi
    vectors=torch.zeros(angles.shape[0], 12, dtype = angles.dtype, device = angles.device)
    vectors[:,0] = 0
    vectors[:,1] = 0
    vectors[:,2] = 1

    # center of detector
    vectors[:,3:6] = 0

    # vector from detector pixel (0,0) to (0,1)
    vectors[:,6] = step
    vectors[:,7] = 0;
    vectors[:,8] = 0;

  # vector from detector pixel (0,0) to (1,0)
    vectors[:,9] = 0
    vectors[:,10] = step
    vectors[:,11] = 0
    vector=vectors[0].detach()

    c1=(angles[:,0]).cos().view(-1,1,1);
    c2=(angles[:,1]).cos().view(-1,1,1);
    c3=(angles[:,2]).cos().view(-1,1,1);

    s1=(angles[:,0]).sin().view(-1,1,1);
    s2=(angles[:,1]).sin().view(-1,1,1);
    s3=(angles[:,2]).sin().view(-1,1,1);

    R = torch.cat([torch.cat([c3*c2*c1-s3*s1, c3*c2*s1 + s3*c1, -c3*s2],dim=2),\
                   torch.cat([-s3*c2*c1-c3*s1,-s3*c2*s1+c3*c1 , s3*s2],dim=2),\
                   torch.cat( [s2*c1,          s2*s1          , c2],dim=2)],dim=1);

    # why transpose here? because transpose is inverse
    # and inverting the matrix means we are applying it to the object rather than the coordinate system
    # if you doubt this, try putting pi/4 for all angles and compare to
    # https://ars.els-cdn.com/content/image/1-s2.0-S1047847705001231-gr1_lrg.jpg
    # the vectors you get out
    vectors[:,0:3] = R.permute(0, 2, 1).matmul(vector[0:3])
    vectors[:,6:9] = R.permute(0, 2, 1).matmul(vector[6:9])
    vectors[:,9:12]= R.permute(0, 2, 1).matmul(vector[9:12])
   
    return vectors


       

