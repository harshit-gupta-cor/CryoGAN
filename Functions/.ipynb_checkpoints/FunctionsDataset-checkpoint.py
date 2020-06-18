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
import numpy as np

import mrcfile
                        
                        

def CoordsLOG(path, micrographNum, threshold=False, N=None, sizeParticle=384):

    coordinatesList=[]
  
    f=open(path+'_autopick.star','r+')
    lines = f.read().replace('\n',',').replace('\t',',').replace(" ",',').split(',')
  
    coordsList=[float(m) for j,m in enumerate(lines) if m is not '' and j>24]
    coordsLOG=np.zeros((len(coordsList)//5, 5))
    for j, val in enumerate(coordsList):
        coordsLOG[j//5, j%5]=val
        
        
        
    coordsLOG=coordsLOG[np.flip(np.argsort(coordsLOG[:,2])),:]
    threshold=np.mean(coordsLOG[:,2])+2*np.std(coordsLOG[:,2])
    f.close()
    
    valid= ((7400-sizeParticle//2 )>coordsLOG[:,0]) .nonzero()[0]   
    coordsLOG=coordsLOG[valid]
    
    valid= (coordsLOG[:,0]>sizeParticle//2 ) .nonzero()[0]   
    coordsLOG=coordsLOG[valid]    
    
    valid= ((7400-sizeParticle//2) >coordsLOG[:,1]) .nonzero()[0]
    coordsLOG=coordsLOG[valid]
    
    valid= (coordsLOG[:,1]>sizeParticle//2 ) .nonzero()[0]   
    coordsLOG=coordsLOG[valid]
    
    if N is not None:
        return coordsLOG[:N]#coords
    elif threshold ==True:
        return coordsLOG[(coordsLOG[:,2]>threshold).nonzero()[0]]#return particles with prob greater than the threshold
    else:
        return coordsLOG



def CoordsEMAN(dir, micrographNum):

    coordinatesList=[]

    f=open(dir+'/EMD-2984_boxes/EMD-2984_'+str(micrographNum).zfill(4)+'.box','r+')
    lines = f.read().replace('\n',',').replace('\t',',').split(',')
    coordsList=[int(m) for m in lines if m is not '']
    coords=np.zeros((len(coordsList)//4, 4))
    for j, val in enumerate(coordsList):
        coords[j//4, j%4]=val
    f.close()
    return coords
  

        
     

def CoordsBackground(pathCoords, pathMicrograph, threshold=False, NumberParticles=10, sizeParticle=384):

    N=sizeParticle
    M=2*N
    CoordsBackground=np.zeros((NumberParticles,2))
    coordsLOG=CoordsLOG(pathCoords, threshold, NumberParticles)   

    
    threshold=np.mean(coordsLOG[:,2])+2*np.std(coordsLOG[:,2])
    
    

    with mrcfile.open(pathMicrograph, permissive=True) as image:

        image=image.data

        isOK = np.ones(image.shape, dtype=bool)
        isOK[:N//2, :] = False
        isOK[-N//2:, :]= False
        isOK[:, :N//2] = False
        isOK[:, -N//2:]= False

    
    BackgroundAreaPercentage=2*NumberParticles*100.0*N**2/(image.shape[0]*image.shape[1])
    for p in coordsLOG:
        
        x = int(p[0])
        y = int(p[1])
        
        isOK[ np.amax(y - M//2, initial=0) :  np.amin(y - M//2 + M, initial=isOK.shape[0]-1), np.amax(x - M//2, initial=0) :  np.amin(x - M//2 + M, initial=isOK.shape[1]-1)] = False # intentional swap, y first!
        areaLeftpercentage=100.0*np.sum(isOK)/(image.shape[0]*image.shape[1])
        if areaLeftpercentage < BackgroundAreaPercentage:
                break
    
    coordsAll = isOK.nonzero()            
    indices = np.random.permutation(len( coordsAll[0] ))[:NumberParticles]
        
    y=coordsAll[0][indices]
    x=coordsAll[1][indices]

    CoordsBackground[:,0]=y
    CoordsBackground[:,1]=x
    ''' 
    for p in range(NumberParticles): 
        coordsAll = isOK.nonzero()
       
        isOK[ np.amax(y - M//2, initial=0) :  np.amin(y -M//2 + M, initial=isOK.shape[0]-1), np.amax(x - M//2, initial=0) :  np.amin(x - M//2 + M, initial=isOK.shape[1]-1)] = False # intentional swap, y first!
    '''  
    return CoordsBackground
         


def CoordsBackground_Betagal_Std(pathCoords, pathMicrograph, threshold=False, NumberParticles=10, sizeParticle=384, downSample=1):

    N=sizeParticle
    M=2*N    

    with mrcfile.open(pathMicrograph, permissive=True) as image:

       
        micrograph=torch.Tensor(image.data).cuda()
    
    template=torch.ones( sizeParticle,sizeParticle).float().cuda()/(sizeParticle**2)
    meanSquare= ConvolveTemplate(micrograph, template, downSample )**2
    Std=ConvolveTemplate(micrograph**2, template, downSample)
    heatmap=Std-meanSquare
    
    
    heatmap=heatmap.cpu().numpy()
    val=np.max(heatmap)
   
    heatmap[:M,:]=val
    heatmap[:,:M]=val

    heatmap[:,-M:]=val
    heatmap[-M:,:]=val
    

    coords=np.zeros((NumberParticles,2))
    for p in range(NumberParticles):

        ind=aminArray(heatmap)
        coords[p,0] = ind[1]
        coords[p,1] = ind[0]
        
        heatmap[ind[0]-sizeParticle//4 : ind[0]+sizeParticle//4 , ind[1]-sizeParticle//4 : ind[1]+sizeParticle//4]=val   
    return coords

def ConvolveTemplate(micrograph, template, downSample=1):
    micrograph=micrograph.unsqueeze(0).unsqueeze(0)
    template=template.unsqueeze(0).unsqueeze(0)
    convolved=torch.nn.functional.conv2d(Down(micrograph, downSample=1), Down(template, downSample=1), padding=(template.shape[-2]//(2*downSample),template.shape[-1]//(2*downSample) ))
    return Up(convolved, downSample).squeeze()


def aminArray(a):
    return np.unravel_index(np.argmin(a, axis=None), a.shape)

def Down(x1, downSample=1):
    n=x1.shape[-1]
    return torch.nn.functional.interpolate(x1, size=[n//downSample,n//downSample])
def Up(x1, downSample=1):
    n=x1.shape[-1]
    return torch.nn.functional.interpolate(x1, size=[downSample*n,n*downSample])