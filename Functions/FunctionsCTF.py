from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch import Tensor

import torch
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

from .FunctionsFourier import *




def CTFGenerator(args, defocusU, defocusV, AngleAstigmatism):
#% f = ctf(n, res, lambda, defocus, Cs, B, alpha)
#% Compute the contrast transfer function corresponding to the resolution
#% res in A/pixel.  Lambda is in A, Cs is in mm,
#% B in A^2 and alpha in radians.  Defocus is in microns.
#% The result is returned in an nxn matrix with h(n/2+1) corresponding
#% to the zero-frequency amplitude.  Thus you must use fftshift() on
#% the result before performing a Fourier transform.  For example,
#% to simulate the effect of the CTF on an image m, do this:
#% fm=cfft2(m);
#% cm=icfft2(fm.*h));
#
#% Cs term fixed.  fs 4 Apr 04
#%
#% The first zero occurs at lambda*defocus*f0^2=1.
#% e.g. when lambda=.025A, defocus=1um, then f0=1/16.
#
#% Usage: lambda is the wavelength determined by the voltage. For example,
#% lambda = EWavelength (300) computes the wavelength in A for 300KV.
#% n is the size of the image, for example n=129. 
#% res = 3 A/pixel.
#% defocus is in micrometers, usually it is in the range of 1 to 4 micrometers.
#% Cs is normally 2.
#% B determines the decay envelope and it can be 0, 10 or 100.
#% alpha is usually set at 0.07.
 
    if args.dataset=='Betagal' or args.dataset=='Betagal-Synthetic':
             
        Resolution=0.637          
        Cs=2.7  
        AmplitudeContrast=0.1
        alpha=0.1         
        PhasePlate=False               
        wavelength=EWavelength(300)[0]   

    Resolution=Resolution* args.DownSampleRate
    
    frequency = 1.0/(args.CTFSize*Resolution)
    
    valueAtNyquist=args.valueAtNyquist 

    decay=(-np.log(valueAtNyquist))**0.5 *2* Resolution
    
    defocusU=defocusU.view(-1,1,1,1)+defocusU.view(-1,1,1,1)*0.0*torch.zeros_like(defocusU.view(-1,1,1,1)).normal_()
    
    defocusV=defocusV.view(-1,1,1,1)+defocusV.view(-1,1,1,1)*0.0*torch.zeros_like(defocusV.view(-1,1,1,1)).normal_()
    
    AngleAstigmatism=AngleAstigmatism.view(-1,1,1,1)    
    
    n2= float(args.CTFSize // 2)
    
    my, mx=torch.meshgrid(torch.arange(-n2,n2+1),torch.arange(-n2,n2+1)) # -n2, ..., n2
    
    mx=mx.to( device=defocusU.device)
    
    my=-my.to( device=defocusU.device)
    
    r2=mx**2+my**2;
    
    angleFrequency=torch.atan2(my,mx)
    
    Elliptical=defocusU*r2 + (defocusV-defocusU)* r2*torch.cos(angleFrequency-AngleAstigmatism).pow(2)
    
    DefocusContribution=np.pi*wavelength*1e4*Elliptical*frequency**2
    
    AbberationContribution=-np.pi/2.0*Cs*(wavelength**3)*1e7*frequency**4 *r2**2;    
   
    Envelope=torch.exp(- (frequency**2) * decay**2 *r2)    
    
    disc=(r2.sqrt().view(-1)<2*n2).view(mx.size()).unsqueeze(0).unsqueeze(0).float()
    
    argument=PhasePlate*np.pi/2+  AbberationContribution+ DefocusContribution
    
    hreal=-( (1-AmplitudeContrast**2)**0.5 * torch.sin(argument)  +  AmplitudeContrast*torch.cos( argument) ); 

    hreal=hreal*Envelope*disc

    hFourier=torch.stack((hreal, torch.zeros_like(hreal)),-1).cuda()
        
    hSpatial=torch.ifft(batch_ifftshift2d(hFourier),2, normalized=False)

    hSpatial=batch_fftshift2d(hSpatial)

    hSpatial=hSpatial.unbind(-1)[0]

    return hFourier, hSpatial


    




    
def EWavelength(kV):
#    % function [lambda sigma]=EWavelength(kV)
#    % Compute the electron wavelength lambda (in angstroms) 
#    % and the interaction parameter sigma (in radians/V.angstrom)
#    % given the electron energy kV in kilovolts.
#    % Uses the relativistic formula 4.3.1.27 in International Tables.  The
#    % interaction parameter is from eqn. (5.6) of Kirkland 2010.
    
    wavelength=12.2639/np.sqrt(kV*1000+0.97845*kV**2);      
    u0=511;  # electron rest energy in kilovolts
    sigma=(2*np.pi/(wavelength*kV*1000))*((u0+kV)/(2*u0+kV));
       
    return wavelength, sigma 



