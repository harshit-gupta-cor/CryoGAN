import torch as th
import torchvision as tv
import Generators_Clean as Generators
import numpy as np
import torch
from torch import nn
import mrcfile
import torch.nn.functional as F

from Functions.FunctionsFourier import *
import cc3d
import numpy as np
from Functions.FunctionsFourier import *
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

from Functions.FunctionsSymmetry import *
from Functions.FunctionsGenerator import *
import cc3d
import numpy as np
from Functions.FunctionsFourier import *
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing


import kornia
import torch
import torch.nn as nn 

from IPython.core.debugger import set_trace
import copy
#=======================================================================
#cryoGenerator
#=======================================================================
class cryoGenerator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self,  args=None):
        from torch.nn import ModuleList

        from torch.nn.functional import interpolate

        super(cryoGenerator, self).__init__()    
        self.G=Generators.ProjectorModule(args=args)

        self.args=args

       
        self.X=th.nn.Parameter(self.G.X.clone().detach()) if self.G.X is not None else th.nn.Parameter(self.G.Xd.clone().detach())   

        self.count=0

        self.Volume=None           
        self.meshgrid=None
        self.PassiveSymmetry=False
        self.iteration=0


        self.VolumeMaskRadius=None      
        self.VolumeMaskRadiusPrev=None

    def forward(self, real_samples, skipDownsampling=False, ChangeAngles=True,  GaussianSigma=None, ratio=0, multipleNoise=False):
        self.iteration=self.iteration+1
        """
        forward pass of the Generator
        """
        projNoisy, projCTF, projClean, self.Volume, Noise=self.G(self.X,real_samples, ChangeAngles=ChangeAngles, ratio=ratio)
        fake_samples=    projNoisy
        return fake_samples, projNoisy, projCTF, projClean, Noise

    def Triangle(self,ratio):
        ratio=1.5*ratio
        if ratio>0.5:
            ratio=1-ratio
        if ratio<0:
            ratio=0
        if ratio>1:
            ratio=1
        return ratio
    def Ramp(self,ratio):
        
        if ratio>0.5:
            ratio=0.5
        if ratio<0:
            ratio=0
        return ratio
    def PositiveMean(self,X):
        mask=(X>0).float()
        return (X*mask).sum()/(mask.sum()+1)
    
    
    
    def Constraint(self, ratio=0):
        if self.args.UseVolumeGenerator==False:
            with torch.no_grad():
                self.VolumeMask(self.VolumeMaskRadius)
                
                if self.args.VolumeMask==False:
                    
                    _, _,self.xx,self.yy,self.zz,self.centrex, self.centrey, self.centrez, self.radius=InitMask(self.args,self.G.Xd, self.G.X, self.VolumeMaskRadius)  
                if self.args.Constraint=='Positive':

                    self.X.data.clamp_(min=-self.args.Value*self.Ramp(ratio)*self.X.data.max() )

                    if self.args.ValueConstraint:
                        self.X.data.clamp_(max=self.args.Value)
                
                elif self.args.Constraint=='Positive-hard':

                    self.X.data.clamp_(min=0)

                    if self.args.ValueConstraint:
                        self.X.data.clamp_(max=self.args.Value)
                
                elif self.args.Constraint=='Positive-Radial':


                    radial=((self.xx-self.centrex)**2+(self.yy-self.centrey)**2+(self.zz-self.centrez) **2).float().sqrt().view(self.X.data.shape[1],self.X.data.shape[2] , self.X.data.shape[3])
                    radialClamps=(self.radius-2-radial).clamp(min=0)/(self.radius-2)
                    radialClamps=radialClamps
                   
                    radialClamps=self.args.Value*self.Ramp(ratio)*self.X.data.max()*radialClamps
                    
                    self.X.data=(self.X.data+radialClamps).clamp_(min=0)-radialClamps
                    
                    radialClamps=(10*(1-radial/self.radius)).clamp_(min=0,max=1)
                    radialClamps=(self.X.data*(radial<0.9*self.radius).float()).max()*radialClamps
                    
                    self.X.data=radialClamps-(radialClamps-self.X.data).clamp_(min=0)
                    
                elif self.args.Constraint=='Negative':
                    self.X.data.clamp_(max=0)
                    if self.args.ValueConstraint:
                        self.X.data.clamp_(min=-1*self.args.Value)


    
    def VolumeMask(self, radius):


        with torch.no_grad():
            if self.args.VolumeMask:

                

                self.VolumeMaskRadiusNew=radius

                if self.VolumeMaskRadiusNew != self.VolumeMaskRadiusPrev:

                    self.mask, self.maskd,self.xx,self.yy,self.zz,self.centrex, self.centrey, self.centrez, self.radius=InitMask(self.args,self.G.Xd, self.G.X, self.VolumeMaskRadiusNew)          
                self.VolumeMaskRadiusPrev=self.VolumeMaskRadiusNew


                if self.args.SymmetryType != 'none':

                    self.X.data = self.X.data*self.maskd.unsqueeze(0)      
                else:

                    self.X.data = self.X.data*self.mask.unsqueeze(0)   
    
   
        
     
    def ExpandVolume(self, X, n, device):
        Y=torch.zeros(n,n,n).to(device)
        x0=(n-X.shape[0])//2
        x1=x0+X.shape[0]
            
        y0=(n-X.shape[1])//2
        y1=y0+X.shape[1]
            
        z0=(n-X.shape[2])//2
        z1=z0+X.shape[2]
            
        Y[x0:x1, y0:y1, z0:z1]=X
        return Y
            

    

#####################################################################       

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np







    
    
    
class DownsamplingCNN(torch.nn.Module):
    def __init__(self, args):
        ''' 6: simple conv network with max pooling'''
        super(DownsamplingCNN, self).__init__()
        self.Fourier=args.FourierDiscriminator
        K = args.num_channel_Discriminator # num channels
        N = args.num_N_Discriminator # penultimate features
        numConvs = args.num_layer_Discriminator


        # first one halves the number of numbers, then multiplies by K
        # interval convolutions, each halves the number of values (because channels double)
     
  
            
        self.convs=nn.ModuleList(
                 [torch.nn.Sequential(
                     torch.nn.Conv2d(2**(i) * K ** (i>0) + 2*self.Fourier*(i==0), 2**(i+1) * K, kernel_size=3, stride=1, padding=1),
                     torch.nn.MaxPool2d(kernel_size=2),
                     torch.nn.LeakyReLU(args.leak_value))
                  for i in range(numConvs) ]
             )




        #todo: have to think about how to handle this
        size = args.ProjectionSize 


        # flatten down to N numbers, then 1 number
       # size=K * size**2 * 2**numConvs / 4**numConvs

        input=torch.zeros(1,1+self.Fourier*2,int(size),int(size) )
        with torch.no_grad():
           
            for conv in self.convs:
                    input = conv(input)

        self.fully=torch.nn.Sequential(
            torch.nn.Linear(np.prod(input.size()),N),
            torch.nn.LeakyReLU(args.leak_value)
            # torch.nn.ReLU()
            )

        self.linear=torch.nn.Linear(N  , 1)
        self.args=args

    def forward(self, input ):
        output = input
        if self.Fourier:
            output=torch.cat([output,SpaceToFourier(output, signal_dim=2).permute(0,4, 2,3,1).squeeze(-1)],1)
        

        for conv in self.convs:
                output = conv(output)

        output = self.fully( output.reshape( output.shape[0], -1) )
        output = self.linear( output )

        return output

