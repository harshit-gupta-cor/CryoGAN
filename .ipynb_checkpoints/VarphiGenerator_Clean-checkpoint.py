from torch.utils.data.dataset import Dataset
from  torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import mrcfile
from torch import Tensor
import os

from Functions.FunctionsDataset import *
from Functions.Functions import *
from Functions.FunctionsCTF import *
from Functions.FunctionsHandle import *
from Functions.FunctionsSymmetry import *
from Functions.FunctionsSaveImage import *
from Functions.FunctionsFourier import *
from IPython.core.debugger import set_trace

import scipy
class VarphiGenerator(Dataset):
   

    def __init__(self,  args=None):
             
        self.args=args   
        self.initEstimated()           
        
      
    def __len__(self):
        print("Varphigenerator data size " +str(self.train_size))
        return  self.train_size
    

    def __getitem__(self, idx):
        with torch.no_grad():
                
                angles= self.AngleGenerator(idx)
                translation= self.TranslationGenerator(  idx)
                defocusU,defocusV,astigmatism=self.DefocusGenerator(idx)                
                Noise=self.NoiseGenerator(idx)
    
                return angles, translation, defocusU, defocusV, astigmatism, Noise
    
    
    def initEstimated(self):
      
       
                   

        if self.args.dataset=='Betagal'or self.args.dataset=='Betagal-Synthetic':

            
                self.BackgroundPath='./Datasets/betagal/Background-384/'
               
              
                print("Background coming from " +self.BackgroundPath)
                
                dir='./Datasets/Betagal-Synthetic/'+self.args.dataset_name+'/'                
                
                if self.args.dataset=='Betagal-Synthetic':
                    if self.args.AlgoType=='generate' :
                        total_images= self.args.DatasetSize
                        
                    else:
                        
                        list=os.listdir(dir  )
                        listmrc=[x for x in list if '.mrc' in x] 
                        self.ImageStack=[dir+x for x in listmrc if 'projNoisy' in x] 
                        self.train_size= len(self.ImageStack)
                        total_images= self.train_size
                else:
            
                
                    
                    total_images=len(os.listdir(self.BackgroundPath))
                    self.train_size=total_images

                
                self.MicrographFromIdx=np.zeros((50000,1))
                dataDir = "/home/jyoo/cryoemfinal/Datasets/betagal/Micrographs"

                counterParticle=-1
                

                for micrographNum in range(1539):
                    
                
                    boxName = "EMD-2984_{:04d}.box".format(micrographNum)

                    numCols = 4
                    boxes = np.fromfile(os.path.join(dataDir, boxName), sep="\t", dtype=np.int).reshape(-1, numCols)
                    for i in range(len(boxes)):
                        counterParticle=counterParticle+1
                        self.MicrographFromIdx[counterParticle,0]=micrographNum
                
                dir= '/home/jyoo/cryoemfinal/Datasets/betagal/'
                pathMicrographCTF=dir+'Micrographs/ctf-betagal-all-3185.txt'
                self.pathCTF=pathMicrographCTF#'./Datasets/ctf-betagal-all-background-6370.txt'
                
                
                
               
                file=open(pathMicrographCTF,'r')
                lines=file.read().split()    
                
                if True:
                    self.EstimatedDefocuses=np.zeros((1539,3))

                    for i,element in enumerate(lines,0):     
                            if i//8==self.EstimatedDefocuses.shape[0]+1:
                                break


                            if self.args.UseEstimatedDefocuses:                    

                                if 5>i%8 >1:
                                     self.EstimatedDefocuses[i//8,i%8 - 2]=float(element)


                    file.close()
                    self.EstimatedDefocuses[:,0]=self.EstimatedDefocuses[:,0]/1e4
                    self.EstimatedDefocuses[:,1]=self.EstimatedDefocuses[:,1]/1e4
                    self.EstimatedDefocuses[:,2]=self.EstimatedDefocuses[:,2]*np.pi/180.0
                    

          
        self.train_size=total_images

            
    def AngleGenerator(self,idx):
        if self.args.UseEstimatedAngles==True:
            angles=  torch.Tensor(self.EstimatedAngles[idx])            
        else:
            angles= RandomAngleGenerator( distType=self.args.AngleDistribution)
        return angles
    
               
    def TranslationGenerator(self,idx):
       
        if self.args.UseEstimatedTranslations==True:
            Translation= torch.Tensor(self.EstimatedTranslations[idx]).long()
        else:               
            TranlsationNormalized=torch.Tensor( 1,2).uniform_().cuda() 
            Translation=(U2T(TranlsationNormalized)*self.args.RawProjectionSize*self.args.TranslationVariance/100.0).float().long()#should be even
        return Translation   
    
    def DefocusGenerator(self,idx):
        if self.args.UseEstimatedDefocuses:
            
            if ('Betagal' in self.args.dataset)  or self.args.dataset=='RiboNew':  
                idx=int(self.MicrographFromIdx[idx,0])            
            
            defocusU= torch.Tensor([self.EstimatedDefocuses[idx,0]]).cuda()
            defocusV=torch.Tensor([self.EstimatedDefocuses[idx,1]]).cuda()
            astigmatism=torch.Tensor([self.EstimatedDefocuses[idx,2]]).cuda()
        else:
             defocusU=self.minDefocus+(self.maxDefocus-self.minDefocus)*torch.zeros(1,1).uniform_()
             defocusV = defocusU.clone().detach()
             astigmatism=torch.zeros_like(defocusU)
                
        return defocusU, defocusV, astigmatism
                
      
                    
    def NoiseGenerator(self, idx):
        self.args.ProjectionSize=int(self.args.ProjectionSize)
            
            
        if self.args.UseEstimatedNoise ==False:
            image=torch.zeros(self.args.ProjectionSize,self.args.ProjectionSize).normal_()

            image=image.unsqueeze(0)
            
      
        else:
            if self.args.dataset=='Betagal' or self.args.dataset=='Betagal-Synthetic' :

                    with mrcfile.open(self.BackgroundPath+str(idx).zfill(6)+".mrc") as m:
                            image=np.array(m.data, dtype=np.float32)                    
                    if self.args.GaussianFilterProjection:                    
                        image=scipy.ndimage.gaussian_filter(image,  self.args.GaussianSigma)
                
                    image=Tensor(image).unsqueeze(0).cuda()
   
       
        downsampling=image.shape[-1]//self.args.RawProjectionSize
        if downsampling>1:
            image=torch.nn.functional.avg_pool2d(image, kernel_size=downsampling, stride=downsampling, padding=0) 
       
        image=(image-image.mean((1,2), keepdim=True)   )/image.std((1,2), keepdim=True)
        
        return image


            
def U2T(x):
    return (x-0.5).sign()*(1-(1-(2*x-1).abs()).sqrt())
            
            
            
            
def RandomAngleGenerator( distType='uniform'):
    if distType=='uniform':
        angles =2*np.pi* torch.empty(3).uniform_()

        angles[1]=np.arccos(angles[1]/(np.pi)-1.0) # to make uniform

       

    elif distType=='cylinder':
        angles= torch.cat([2*np.pi*torch.empty(1).uniform_(),\
            0.5*np.pi*(torch.randint(low=0, high=2, size=(1,), dtype=torch.float)) + \
            0.5*np.pi*(torch.randint(low=0, high=2,size=(1,),dtype=torch.float)),\
            2*np.pi*torch.empty(1).uniform_()])
        
    elif distType=='cylindernoisy':
        angles= torch.cat([2*np.pi*torch.empty(1).uniform_(),\
            0.5*np.pi*(torch.randint(low=0, high=2, size=(1,), dtype=torch.float)) + \
            0.5*np.pi*(torch.randint(low=0, high=2,size=(1,),dtype=torch.float)),\
            2*np.pi*torch.empty(1).uniform_()])   
        if (angles[1] -0.5*np.pi).abs() > 0.1:
            angles[1]=angles[1]+(0.07*np.pi*2*(torch.empty(1).uniform_()-0.5))
        else:
            angles[1]=angles[1]+(0.05*np.pi*2*(torch.empty(1).uniform_()-0.5))
            
        
    return angles

    

            
def EstmatedNoise(args):

    try:
        Noise,imageNum = args.EstimatedNoiseIterator.next()
    except StopIteration: # reshuffle the dataset

        args.EstimatedNoiseIterator = iter(torch.utils.data.DataLoader(args.EstimatedNoiseDataset,batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)    )
        Noise, imageNum = args.EstimatedNoiseIterator.next()

    return  Noise.to('cuda', non_blocking=True), imageNum, args






    
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

