""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""

import os
import time
import copy
import numpy as np
import torch as th
import astra
import torch
from torch import Tensor
import sys
import shutil
from shutil import copyfile
from Networks_Clean import *
from Functions.FunctionsCTF import *
from Functions.FunctionsFourier import *
from Functions.Functions import *
import mrcfile
import scipy.io 
import libs.plot
# temp timing
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
from IPython.core.debugger import set_trace

plt.switch_backend('agg')
 
 
        
        
class GenerateData:
  

    def __init__(self, device=th.device("cpu"), args=None):
      
   
        from torch.optim import Adam
        from torch.nn import DataParallel
        
        self.args=args
        
        self.gen = cryoGenerator( args=self.args).to(device)
        self.initGT(device)
    def paths(self):
        OUTPUT_PATH= './'+'Datasets/'
        if os.path.exists(OUTPUT_PATH)==False:
            os.mkdir(OUTPUT_PATH)


        OUTPUT_PATH = OUTPUT_PATH+self.args.dataset # output path where result (.e.g drawing images, cost, chart) will be stored

        if os.path.exists(OUTPUT_PATH)==False:
            os.mkdir(OUTPUT_PATH)

       
        if self.args.name is not None:
                name = self.args.dataset_name 
        else:
                name =''
        OUTPUT_PATH= os.path.join(OUTPUT_PATH, name )

        if os.path.exists(OUTPUT_PATH) ==False :
            print("Making new directory.")
            os.mkdir(OUTPUT_PATH)   
        else:
            print("Old directory exists. Deleting and making a new one")
            shutil.rmtree(OUTPUT_PATH)
            os.mkdir(OUTPUT_PATH)  
        print('Saving files in ' + OUTPUT_PATH)
        return OUTPUT_PATH 
    
    
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
            
    def initGT(self, device):
       
        
        if 'Ribo' in self.args.dataset:
             self.GroundTruth=torch.Tensor( mrcfile.open('./Datasets/Ribo/emd_2660.mrc').data).to(device)
             self.gen.X.data=self.GroundTruth
        elif 'Betagal' in self.args.dataset:
            print("Using 2.5 A molmap")
            fittedBetagal=torch.Tensor( mrcfile.open('./Datasets/Betagal-Synthetic/fitted_betagal_2.5A.mrc').data).to(device)
            fittedBetagal=torch.nn.functional.avg_pool3d(fittedBetagal.unsqueeze(0).unsqueeze(0), kernel_size=self.args.DownSampleRate, stride=self.args.DownSampleRate, padding=0).squeeze()
           
            n=self.args.VolumeSize
            
            GroundTruth= self.ExpandVolume( fittedBetagal, n, device)
            self.gen.X.data=GroundTruth.unsqueeze(0)
            self.GroundTruth=self.gen.X.data
         
                
                
        elif 'ABC' in self.args.dataset:
            n=self.args.VolumeSize
            
            if self.args.VolumeNumbers==1:
                vol=torch.Tensor( mrcfile.open('./Datasets/ABC-Synthetic/fitted_ABC.mrc').data).to(device)
                self.gen.X.data=self.ExpandVolume( vol, n, device).unsqueeze(0)
                
            else:    
                for i in range(self.args.VolumeNumbers):
                    num=4773+2*i
                    vol=torch.Tensor( mrcfile.open('./Datasets/ABC/fitted_'+str(num)+'.mrc').data).to(device)/4
                    self.gen.X.data[i]= self.ExpandVolume( vol, n, device).unsqueeze(0)

         
            self.GroundTruth=self.gen.X.data
            
        elif 'proteasome' in self.args.dataset:
            n=self.args.VolumeSize
           
                
            vol=torch.Tensor( mrcfile.open('./Datasets/proteasome-Synthetic/fitted_proteasome.mrc').data).to(device)
            self.gen.X.data[0]= self.ExpandVolume( vol, n, device)
            self.GroundTruth=self.gen.X.data
        
        elif 'serotonin' in self.args.dataset:
            n=self.args.VolumeSize
           
                
            vol=torch.Tensor( mrcfile.open('./Datasets/serotonin-Synthetic/fitted_serotonin.mrc').data).to(device)
            self.gen.X.data[0]= self.ExpandVolume( vol, n, device)
            self.GroundTruth=self.gen.X.data
            
            
        if self.args.VolumeDomain =='fourier' and self.args.FourierProjector==True:
            self.gen.X.data=fftshift(torch.rfft(ifftshift(self.GroundTruth, mode='real', signal_dim=3), 3, onesided=False),mode='complex', signal_dim=3)
            print("Fourier projector")
        
            
    def SaveVolumeSlices(self, OUTPUT_PATH, iteration)  :  
        
        if self.args.VolumeDomain=='fourier':
            volumeReal, volumeImag= torch.unbind(self.gen.X.data, -1)
            volumeFourier =(volumeReal**2+volumeImag**2+1e-12).sqrt().log()
            volume=FourierToSpace(self.gen.Volume)
        else:
            volume=self.gen.X.data
            volumeFourier=None
        symmetryAxis=None
        
        if self.args.SymmetryType !='none':
            symmetryAxis=np.argmax(self.gen.X.shape)
            volume=torch.transpose(volume, 1, symmetryAxis)
        
        i= np.random.randint(0, volume.shape[0])
        volumeSingle =volume[i]
            
        save_fig(volumeSingle.unsqueeze(1).cpu().data, 
                OUTPUT_PATH, 'volume_'+str(i+1), iteration=iteration, 
                 Title='volume' + str(iteration), doCurrent = True)
        if volumeFourier is not None:
             if symmetryAxis is not None:
                 volumeFourier=torch.transpose(volumeFourier, 0, symmetryAxis)
             
             save_fig(volumeFourier[i].unsqueeze(1).cpu().data, 
                OUTPUT_PATH, 'volumeFourier_'+str(i+1), iteration=iteration, 
                 Title='volumeFourier' + str(iteration), doCurrent = True)

        
    def generate(self ):
        epochs=self.args.DatasetSize
        OUTPUT_PATH=self.paths()
        sample_dir=OUTPUT_PATH
        
        #self.gen.Constraint()
        with open(OUTPUT_PATH + '/config.cfg', 'w') as fp:
            self.args.config.write(fp)
        
        
        if 'Ribo' in self.args.dataset :
            
            val=0.19
            self.gen.X.data=(self.gen.X.data>val).float()*self.gen.X.data
          
        with torch.no_grad():
            for i, vol in enumerate(self.gen.X.data):
                with mrcfile.new(os.path.join(sample_dir, 'GroundTruth_'+str(i)+'.mrc'), overwrite=True) as m:                   
                    m.set_data(self.gen.X[i].data.cpu().numpy())   
                    
        self.SaveVolumeSlices(OUTPUT_PATH, str(0))

        self.gen.G.scalar.data[1]=-2
        
        fscNoise=None
        fscCTF=None
        fscClean=None 
        fscRatio=None
        
        
        for i in range(epochs//self.args.BATCH_SIZE):
            
            with torch.no_grad():
                if self.args.dataset=='bs':
                    alpha=np.pi/3*(2*np.random.random(1,)-1)
                    self.gen.X.data=BS(self.args.VolumeSize, alpha, binary=True, device='cuda:0').float().unsqueeze(0)
                fake_samples, projNoisy, projCTF, projClean, Noise = self.gen(None)
                
                
                for j,projNoisySingle in enumerate(projNoisy):
                    num=j+i*self.args.BATCH_SIZE
                    with mrcfile.new(os.path.join(sample_dir, 'projNoisy'+str(num).zfill(6)+'.mrc'), overwrite=True) as m:                   
                        m.set_data(projNoisySingle.squeeze(0).data.cpu().numpy()) 

                ''' 
                with mrcfile.new(os.path.join(sample_dir, 'projCTF'+str(i).zfill(6)+'.mrc'), overwrite=True) as m:                   
                    m.set_data(projCTF.squeeze(0).squeeze(0).data.cpu().numpy()) 

                with mrcfile.new(os.path.join(sample_dir, 'projClean'+str(i).zfill(6)+'.mrc'), overwrite=True) as m:                   
                    m.set_data(projClean.squeeze(0).squeeze(0).data.cpu().numpy()) 
                '''

                
                save= i % (epochs//(10*self.args.BATCH_SIZE))==0
                fscNoise, fscCTF, fscClean,fscRatio= SaveRadial(Noise, projCTF, projClean, OUTPUT_PATH, fscNoise=fscNoise, fscCTF=fscCTF, fscClean=fscClean, fscRatio=fscRatio, save=save)
                
                    
                if save:
                    print("Done "+str(num).zfill(6))
                   
                    if self.args.dataset=='bs':
                        for numvol, vol in enumerate(self.gen.X.data):
                            with mrcfile.new(os.path.join(sample_dir, 'GroundTruth_'+str(num)+'_'+str(i)+'.mrc'), overwrite=True) as m:
                                    m.set_data(self.gen.X[numvol].data.cpu().numpy())   
                    save_fig_double(projClean.cpu().data,projCTF.cpu().data,
                            OUTPUT_PATH, "ProjClean", iteration=str(i),
                            Title1='ProjClean', Title2='ProjCTF' + str(i),
                            doCurrent=True, sameColorbar=False)

                    save_fig_double(projCTF.cpu().data,projNoisy.cpu().data,
                            OUTPUT_PATH, "ProjNoisy", iteration=str(i),
                            Title1='ProjCTF', Title2='ProjNoisy' + str(i),
                            doCurrent=True, sameColorbar=False)

   
def SaveRadial(Noise, projCTF, projClean, OUTPUT_PATH, fscNoise=None, fscCTF=None, fscClean=None, fscRatio=None, save=False):
    
    fscNoiseCurrent=RadiallyAverageFourierTransform(Noise)
    fscCTFCurrent=RadiallyAverageFourierTransform(projCTF)
    fscCleanCurrent=RadiallyAverageFourierTransform(projClean)
    # B x FSC
    fscNoiseCurrent[:,0]=fscNoiseCurrent[:,1]
    fscRatioCurrent=fscCTFCurrent/fscNoiseCurrent
    
    if fscNoise is not None:
        fscNoise=fscNoise+fscNoiseCurrent.sum(0)
        fscCTF=fscCTF+fscCTFCurrent.sum(0)
        fscClean=fscClean+fscCleanCurrent.sum(0)
        fscRatio=fscRatio+fscRatioCurrent.sum(0)
    else:
        fscNoise=fscNoiseCurrent.sum(0)
        fscCTF=fscCTFCurrent.sum(0)
        fscClean=fscCleanCurrent.sum(0)
        fscRatio=fscRatioCurrent.sum(0)
        

        
    if save:
        fig=plt.figure(4)
        plt.subplot(221)
        plt.title("Noise")
        plt.plot(fscNoise)

        plt.subplot(222)
        plt.title("CTF")
        plt.plot(fscCTF)

        plt.subplot(223)
        plt.title("Clean")
        plt.plot(fscClean)

        plt.subplot(224)
        plt.title("CTF/Noisy")
        plt.plot(fscRatio)
    
        plt.savefig(OUTPUT_PATH+"/RadialAverage.pdf")
        plt.close(fig)
        
        fig=plt.figure(4)
        plt.subplot(221)
        plt.title("Noise")
        plt.plot( (fscNoise+1e-12).log10())

        plt.subplot(222)
        plt.title("CTF")
        plt.plot((fscCTF+1e-12).log10())

        plt.subplot(223)
        plt.title("Clean")
        plt.plot((fscClean+1e-12).log10())

        plt.subplot(224)
        plt.title("CTF/Noisy")
        plt.plot((fscRatio+1e-12).log10())
    
        plt.savefig(OUTPUT_PATH+"/RadialAverageLog.pdf")
        plt.close(fig)
        
        
        fig=plt.figure(4)
        plt.subplot(221)
        plt.title("Noise")
        plt.plot(fscNoiseCurrent[:4].t())

        plt.subplot(222)
        plt.title("CTF")
        plt.plot(fscCTFCurrent[:4].t())

        plt.subplot(223)
        plt.title("Clean")
        plt.plot(fscCleanCurrent[:4].t())

        plt.subplot(224)
        plt.title("CTF/Noisy")
        plt.plot(fscRatioCurrent[:4].t())
    
        plt.savefig(OUTPUT_PATH+"/RadialAverageCurrent.pdf")
        plt.close(fig)
        
        
        fig=plt.figure(4)
        plt.subplot(221)
        plt.title("Noise")
        plt.plot( (fscNoiseCurrent[:4].t()+1e-12).log())

        plt.subplot(222)
        plt.title("CTF")
        plt.plot((fscCTFCurrent[:4].t()+1e-12).log())

        plt.subplot(223)
        plt.title("Clean")
        plt.plot((fscCleanCurrent[:4].t()+1e-12).log())

        plt.subplot(224)
        plt.title("CTF/Noisy")
        plt.plot((fscRatioCurrent[:4].t()+1e-12).log())
    
        plt.savefig(OUTPUT_PATH+"/RadialAverageCurrentLog.pdf")
        plt.close(fig)
        
        
        
        
        fig=plt.figure(4)
        plt.subplot(221)
        plt.title("1")
        plt.plot( normalize( (fscNoiseCurrent[0]+1e-12).log() ) )
        plt.plot( normalize( (fscCTFCurrent[0]+1e-12).log() ) )

        plt.subplot(222)
        plt.title("2")
        plt.plot( normalize( (fscNoiseCurrent[1]+1e-12).log() ) )
        plt.plot( normalize( (fscCTFCurrent[1]+1e-12).log() ) )
        
        plt.subplot(223)
        plt.title("3")
        plt.plot( normalize( (fscNoiseCurrent[2]+1e-12).log() ) )
        plt.plot( normalize( (fscCTFCurrent[2]+1e-12).log() ) )
        
        plt.subplot(224)
        plt.title("4")
        plt.plot( normalize( (fscNoiseCurrent[3]+1e-12).log() ) )
        plt.plot( normalize( (fscCTFCurrent[3]+1e-12).log() ) )
    
        plt.savefig(OUTPUT_PATH+"/RadialAverageCurrentNoiseProjLog.pdf")
        plt.close(fig)
    return fscNoise, fscCTF, fscClean,fscRatio
   
def normalize(x):
    return (x-x.min())/(x[x.shape[-1]//2:].max()-x.min())
                         
import torch
import mrcfile
import numpy as np
import math

def sphere(V, x, y, z, r):
    '''V: torch 3D tensor representing the volume
       x,y: center coordinate in V
       r: radius
       return: V with a sphere added'''
    d = torch.arange(0, V.size(0)).to(V.device)
    i, j, k = torch.meshgrid(d,d,d)
    i = i.float()
    j = j.float()
    k = k.float()
    sphere = ((i-x)**2 + (j-y)**2 + (k-z)**2 <= r**2).float()
    
    return V+sphere

def cylinder(V, x, y, z, a, l, r):
    '''
        V: same
        x,y,z: center of top-left end of cylinder
        a: angle of normal vector wrt z axis (a<=pi/2), assum angle wrt x axis = pi/2 (on yz plane)
        l: length
        r: radius'''
    d = torch.arange(0, V.size(0)).to(V.device)
    i, j, k = torch.meshgrid(d,d,d)
    i = i.float()
    j = j.float()
    k = k.float()
    dx = 0
    dy = math.sin(a)
    dz = math.cos(a)
    cylinder = ((i-x)*dx + (j-y)*dy + (k-z)*dz <= l) \
               & ((i-x)*dx + (j-y)*dy + (k-z)*dz >= 0) \
               & ((i-x)**2 + (j-y)**2 + (k-z)**2 - ((i-x)*dx + (j-y)*dy + (k-z)*dz)**2 <= r)
               
   
    return V+cylinder.float()

# def BS(N, alpha, binary=True, device='cpu'):
#     V = torch.zeros([N, N, N]).to(device)
#     if alpha<np.pi//2:
#         V = sphere(V, N/4, N/4, N/4, N//16)
#         V = cylinder(V, N/4, N/4, N/4, 0, N//2, N//8)
#         V = cylinder(V, N/4, N/4, N/4, alpha,  N//2, N//8)
#     else:
#         alpha=alpha-np.pi//2
#         V = sphere(V, N/4, N/4, 3*N/4, N//16)
#         V = cylinder(V, N/4, N/4, N/4, 0, N//2, N//8)
#         V = cylinder(V, N/4, N/4, N/4, alpha,  N//2, N//8)
#     if binary:
#          V = V> 0
#     return V.float()
        
def BS(N, alpha, binary=True, device='cpu'):
    V = torch.zeros([N, N, N]).to(device)
    sphereRadius=N//16
    
    length=N//2
    
    if alpha>0:
        centre=N/2-sphereRadius
        V = sphere(V, centre, centre, centre, sphereRadius)
        V = cylinder(V, centre, centre, centre, 0, length, N//8)
        V = cylinder(V, centre, centre, centre, alpha, length, N//8)
    else:
        
        centre=N/2
        V = sphere(V, centre, centre, centre+sphereRadius, sphereRadius)
        V = cylinder(V, centre, centre, centre+sphereRadius-length, 0, length, N//8)
        V = cylinder(V, centre, centre, centre+sphereRadius-length, alpha, length, N//8)
        
    
    if binary:
         V = V> 0
    return V.float()                        
                            
                       