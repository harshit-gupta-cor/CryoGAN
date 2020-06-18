from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch import Tensor

import torch as th
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
from .FunctionsFourier import *
import torch
from IPython.core.debugger import set_trace
import kornia
    
import torch.nn.functional as F
import numpy as np

def weights_init(m, args):
    
    if isinstance(m, nn.Conv2d):
        
        if m.weight is not None:            
            init.kaiming_normal_(m.weight, a = args.leak_value)           
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
        
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.kaiming_normal_(m.weight, a = args.leak_value)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
            
            
def dataProducer(dataiter, dataloader, device):
    X = next(dataiter, None)
    
    if X is None:
        dataiter = iter(dataloader)           
        X = dataiter.next()
    return X.to(device)

                               

def ProjectionMask(x, ProjectionMask=None):
    return ProjectionMask*x


    
    
def stable_gradient_penalty( dis, real_samps, fake_samps, gamma_gradient_penalty):
   
  
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        op = dis(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradients = th.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=th.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1)  +1e-12     )

            # Return gradient penalty
        return  ((gradients_norm/gamma_gradient_penalty - 1) ** 2).mean()

def dis_loss( dis, real_samps, fake_samps,  reg_lambda, lambda_drift, gamma_gradient_penalty ):
        # define the (Wasserstein) loss
        
        fake_out = dis(fake_samps)
        
        real_out = dis(real_samps)
        
        wassloss = th.mean(fake_out) - th.mean(real_out)

        gp = reg_lambda*stable_gradient_penalty(dis,real_samps, fake_samps,gamma_gradient_penalty )
        
        loss = wassloss+ gp+lambda_drift*th.mean(real_out.pow(2))

        return loss, -1*wassloss.detach().item()

def gen_loss(dis, fake_samps):
        # calculate the WGAN loss for generator
        loss = -th.mean(dis(fake_samps))

        return loss
