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
import os

## temp
from timeit import default_timer as timer
from datetime import timedelta
##

imshowParams = {
    'cmap' : 'gray',
    'interpolation' : 'none'
    }

ext = '.pdf'
savefigParams = {
    'bbox': 'tight'
    }


def imshow_version(x, mode='all',padding=0):
    if mode=='all':
        
        y=x[:16].squeeze(1)    
        im=make_grid(y.view(-1,1,x.size()[-2],x.size()[-1]), 4,padding=padding)
      
  
        
    return np.transpose(im.numpy(),(1,2,0))[:,:,0]

def save_fig(x, path, name, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False):
    
    if Title is None:
        Title = name
    
   
    fig =plt.figure(1,figsize= (20,20))
   
    im=np.transpose(make_grid(x,nrow=int(np.sqrt(x.size()[0])),padding=0,scale_each=scaleEach,normalize=scaleEach).numpy(),(1,2,0))[:,:,0]
    plt.imshow(im, **imshowParams)
    
    if not scaleEach: ## don't draw colorbar if each panel is caled separately
        plt.colorbar(fraction=0.046, pad=0.04)

    if torch.any(torch.isnan(x)):
        Title += " *******WARNING, NaN's detected********"
    
    plt.title(Title)

    
    if doCurrent:
        fig.savefig(os.path.join(path, 'current-' + name + ext), **savefigParams )

    if iteration is not None:
        name = iteration + "-" + name
        
    fig.savefig(os.path.join(path, name + ext), **savefigParams )


    # if you want raw data saved:
    # save_image(x, os.path.join(path, name + '.png'), normalize=True)

    plt.close(fig)


def save_fig_single(x, path, name, nrow=None, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False):

    Title = name if Title is None else Title
    nrow=int(np.sqrt(x.size()[0])) if nrow is None else nrow
    
    fig =plt.figure(1,figsize= (10*nrow,10*x.shape[0]//nrow))
   
    im=np.transpose(make_grid(x,nrow,padding=int(0.05*x.shape[-1]),scale_each=scaleEach,normalize=scaleEach).numpy(),(1,2,0))[:,:,0]
    plt.imshow(im, **imshowParams)
    plt.axis('off')
   
    if torch.any(torch.isnan(x)):
        Title += " *******WARNING, NaN's detected********"
    
    plt.title(Title)

    
    if doCurrent:
        fig.savefig(os.path.join(path, 'current-' + name + ext), **savefigParams )

    if iteration is not None:
        name = iteration + "-" + name
        
    fig.savefig(os.path.join(path, name + ext), **savefigParams )


    # if you want raw data saved:
    # save_image(x, os.path.join(path, name + '.png'), normalize=True)

    plt.close(fig)   
    
def save_fig_single_separate(x, path, name, nrow=None, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False, vminvalue=None, vmaxvalue=None):

    Title = name if Title is None else Title
    nrow=int(np.sqrt(x.size()[0])) if nrow is None else nrow
    
    fig =plt.figure(1)#,figsize= (10*nrow,10*x.shape[0]//nrow))
   
    x=x.numpy()
    #string=str(x.shape[0]//nrow)+str(nrow)
    
    for i, im in enumerate(x):
        
        #plt.subplot(string+str(i))
        if vminvalue is not None:
            vmin=vminvalue
            vmax=vmaxvalue
            plt.imshow(im.squeeze(), **imshowParams, vmin=vmin, vmax=vmax)
        else:
            plt.imshow(im.squeeze(), **imshowParams)
        plt.axis('off')
   
  
        
    fig.savefig(os.path.join(path, name + ext), **savefigParams,bbox_inches='tight' ,pad_inches=-0.1)
    plt.close(fig)    


    # if you want raw data saved:
    # save_image(x, os.path.join(path, name + '.png'), normalize=True)

                   

def save_fig_double(x, y, path, name, iteration=None, doCurrent=False, \
                    figshow=False, Title1= None,Title2= None, sameColorbar=False, mask=None ):

    fig =plt.figure(2,figsize= (40,20))
    with torch.no_grad():
        if mask is not None:
            xNonZero=x+(mask.cpu()==0).float()*(x.max()+x.min())/2
            yNonZero=y+(mask.cpu()==0).float()*(y.max()+y.min())/2
            
        else:
            xNonZero=x
            yNonZero=y
        
        vmax = np.max([torch.max(xNonZero), torch.max(yNonZero)])
        vmin = np.min([torch.min(xNonZero), torch.min(yNonZero)])

    
    plt.subplot(1,2,1)
    if sameColorbar:
        plt.imshow(imshow_version(x,padding=0), vmin=vmin, vmax=vmax, **imshowParams)
    else:
        plt.imshow(imshow_version(x,padding=0),vmin=torch.min(xNonZero), vmax=torch.max(xNonZero), **imshowParams)

    if torch.any(torch.isnan(x)):
        Title1 += " *******WARNING, NaN's detected********"
    plt.title(Title1)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1,2,2)
    if sameColorbar:
        plt.imshow(imshow_version(y,padding=0), vmin=vmin, vmax=vmax, **imshowParams)
    else:
        plt.imshow(imshow_version(y,padding=0), vmin=torch.min(yNonZero), vmax=torch.max(yNonZero), **imshowParams)

    if torch.any(torch.isnan(y)):
        Title2 += " *******WARNING, NaN's detected********"
    plt.title(Title2)
    plt.colorbar(fraction=0.046, pad=0.04)

    
    if doCurrent:
        fig.savefig(os.path.join(path, 'current-' + name + ext),  **savefigParams )

    if iteration is not None:
        name = iteration + "-" + name
        
    fig.savefig(os.path.join(path, name + ext),  **savefigParams )
    

    plt.close(fig)


