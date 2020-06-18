import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import mrcfile
import scipy
from scipy import ndimage
import torch
from scipy import misc

def CoordsBackground_Betagal_Std( Micrograph, threshold=False, NumberParticles=10, sizeParticle=150, downSample=1):

    N=sizeParticle
    M=2*N    


       
    micrograph=torch.Tensor(Micrograph).cuda()
    
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
    if downSample>1:
        n=x1.shape[-1]
        return  torch.nn.functional.interpolate(x1, size=[n//downSample,n//downSample])
    else:
        return x1
    
def Up(x1, downSample=1):
    if downSample>1:
        n=x1.shape[-1]
        return torch.nn.functional.interpolate(x1, size=[downSample*n,n*downSample])
    else:
        return x1

    

plt.rcParams['image.cmap']='gray'

plt.close("all")

dataDir = "/home/jyoo/cryoemfinal/Datasets/betagal/Micrographs"
pathSaveParticles= "./Datasets/betagal/Particles"
pathSaveBackground="./Datasets/betagal/Background"
counterParticle=-1
counterBackground=-1

for micrographNum in range(0,1539):
    print(micrographNum, counterParticle,counterBackground )
    mrcName = "EMD-2984_{:04d}.mrc".format(micrographNum)
    boxName = "EMD-2984_{:04d}.box".format(micrographNum)

    numCols = 4
    boxes = np.fromfile(os.path.join(dataDir, boxName), sep="\t", dtype=np.int).reshape(-1, numCols)

    with mrcfile.open(os.path.join(dataDir, mrcName), permissive=True) as mrc:
        im = mrc.data
   
    

    i = 1
    j = 1-i
    for boxInd in range(len(boxes)):
        counterParticle=counterParticle+1
        box = boxes[boxInd]/2
        x0=box[i]+box[2]//4
        x1=box[i] + 3*box[2]//4
        
        y0=box[j]+box[3]//4
        y1=box[j] + 3*box[3]//4
        #xDS = imSmoothDS[x0:x1, y0:y1]
        x=im[int(2*x0):int(2*x1), int(2*y0):int(2*y1)]
        with mrcfile.new(os.path.join(pathSaveParticles, str(counterParticle).zfill(6)+".mrc"), overwrite=True) as m:
            m.set_data(x)
        
    N=384
    
    #smooth the micrograph for better statisctics
    imSmooth = scipy.ndimage.gaussian_filter(im, 20)
    imSmoothDS=Down(torch.Tensor(imSmooth).unsqueeze(0).unsqueeze(0).cuda(),  downSample=2).squeeze().cpu().numpy()
    coordsBackground=CoordsBackground_Betagal_Std( imSmoothDS, threshold=False, NumberParticles=len(boxes), sizeParticle=N, downSample=1)

    for i in range(len(coordsBackground)):
        counterBackground=counterBackground+1
        x = int(coordsBackground[i,0])
        y = int(coordsBackground[i,1])
        
        x0=y-N//2
        x1=y+N//2
        
        y0=x-N//2
        y1=x+N//2
        #imBack = imSmoothDS[x0:x1, y0: y1]    
        imBack = im[int(2*x0):int(2*x1), int(2*y0):int(2*y1)]
  
        with mrcfile.new(os.path.join(pathSaveBackground, str(counterBackground).zfill(6)+".mrc"), overwrite=True) as m:
            m.set_data(imBack)
    

