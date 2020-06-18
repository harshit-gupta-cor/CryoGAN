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
from Networks import *
from Functions.FunctionsCTF import *
from Functions.FunctionsFourier import *
from Functions.Functions import *
from Functions.FunctionsSaveImage import *
import mrcfile
import scipy.io 
import libs.plot
# temp timing
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import itertools
plt.switch_backend('agg')
 
 
        
        
class CreateFigure:
  

    def __init__(self, depth=7, latent_size=32, learning_rate=0.001, beta_1=0.5,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", use_ema=True, ema_decay=0.999,
                 device=th.device("cpu"), args=None):
      
   
        from torch.optim import Adam
        from torch.nn import DataParallel
        
        self.args=args
        self.args.BATCH_SIZE=3
        self.args.Batch_Size=3
        
        self.args.SymmetryType='none'
        self.gen = cryoGenerator(depth, latent_size, use_eql=use_eql, args=args).to(device)
        self.device=device
    def GroundTruthPath(self):
        OUTPUT_PATH= './'+'Datasets/'

        OUTPUT_PATH = OUTPUT_PATH+self.args.dataset       
        name = self.args.name if self.args.name is not None else ''        
        OUTPUT_PATH= os.path.join(OUTPUT_PATH, name ) 
        OUTPUT_PATH='./Results/Results/RiboNew/NoiseAdder-even-average-2019-11-05T20-50/
        return OUTPUT_PATH
    
    def ReconstructionPath(self):
        OUTPUT_PATH= './'+'Results/'
        OUTPUT_PATH = OUTPUT_PATH+self.args.dataset+'/'
        name = self.args.name+'-' if self.args.name is not None else ''        
        lists=[ OUTPUT_PATH+folderName for folderName in os.listdir(OUTPUT_PATH) if name in folderName ]
       
        lists.sort(key=lambda x: os.path.getctime(x))
        
        
        #OUTPUT_PATH='./Results/NoiseAdder-odd-average-2019-11-05T20-50/'#lists[-1]
        OUTPUT_PATH='./Results/Results/RiboNew/NoiseAdder-odd-average-2019-11-05T20-50/'#lists[-1]
        return OUTPUT_PATH
     
        
    def ReconstructedSequence(self, OUTPUT_PATH):
        
        lists=[ os.path.join(OUTPUT_PATH,fileName) for fileName in os.listdir(OUTPUT_PATH) if ('.mrc' in fileName) and ('volume'  not in fileName)]
      
        lists.sort(key=lambda x: os.path.getctime(x))
        
        return lists
    
    def FigurePath(self):
        OUTPUT_PATH= './'+'Figures/'
        if os.path.exists(OUTPUT_PATH)==False:
            os.mkdir(OUTPUT_PATH)


        OUTPUT_PATH = OUTPUT_PATH+self.args.dataset 

        if os.path.exists(OUTPUT_PATH)==False:
            os.mkdir(OUTPUT_PATH)
            
        name = self.args.name if self.args.name is not None else ''
                
        
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
    
    def saveArrayAsBinaryFile(self,filename, array):
        f=open(filename, 'w+b')
        x=np.array(array, dtype=np.float32)
        x.tofile(f)
        f.close()
        
    def registerVolumes(self, GroundTruth,Reconstruction, VolumeMask, FigurePath , StableVolume='GroundTruth'):
        
        
        if StableVolume != 'GroundTruth':
            StableVolume ='Reconstruction'
        StableVolumeArray=GroundTruth if StableVolume=='GroundTruth' else Reconstruction
        UnstableVolumeArray=Reconstruction if StableVolume=='GroundTruth' else GroundTruth
        
        
        registrationDir='/home/hgupta/cryoemfinal/regDemo/'
        self.saveArrayAsBinaryFile(registrationDir + 'StableVolume',StableVolumeArray)
        self.saveArrayAsBinaryFile(registrationDir + 'UnstableVolume',UnstableVolumeArray)
        self.saveArrayAsBinaryFile(registrationDir + 'VolumeMask',VolumeMask)
        
        
        current_dir=os.getcwd()
        os.chdir(registrationDir)
        os.system("gcc  main.c  BsplnTrf.c  convolve.c     pyrFilt.c   quant.c reg1.c reg3.c  svdcmp.c BsplnWgt.c  getPut.c    message.c  pyrGetSz.c  reg0.c   reg2.c  regFlt3d.c -lm -o main")
        os.system("./main")
        os.chdir(current_dir)
        
        registeredReconstruction=np.fromfile('/home/hgupta/cryoem/registeredVolume', dtype=np.float32)
        registeredReconstruction=registeredReconstruction.reshape(Reconstruction.shape)
        
       
       
                
        return registeredReconstruction

     
    def SaveVolumeSlices(self, OUTPUT_PATH, iteration)  :  
        with torch.no_grad():
            if self.args.SymmetryType =='none':
                volume=self.gen.X
            else:
                symmetryAxis=np.argmax(self.gen.X.shape)
                volume=torch.transpose(self.gen.X, 0, symmetryAxis)

            save_fig(volume.unsqueeze(1).cpu().data, 
                    OUTPUT_PATH, 'GroundTruthSlices', iteration=iteration, 
                     Title='volume' + str(iteration), doCurrent = True)   
    
    
    
    def maskShell(self, xx, yy, zz, r, w, size):
        mask=((xx**2+yy**2+zz**2).sqrt()-r).abs()< w
        return mask.cuda()


    def fsc(self, x1,x2, VolumeMask, size, resolution=0.637):
        # compute fft
        with torch.no_grad():
            device=x1.device
            fftx1real, fftx1imag = torch.unbind(SpaceToFourier(x1*VolumeMask, signal_dim=3), -1 )
            fftx2real, fftx2imag  = torch.unbind(SpaceToFourier(x2*VolumeMask, signal_dim=3), -1) 

            # set coordinates
            n1=x1.shape[0]//2
            n2=x1.shape[0]//2
            n3=x1.shape[0]//2
            xx=torch.arange(-n1, n1).to(device).float()/n1
            yy=torch.arange(-n2, n2).to(device).float()/n2
            zz=torch.arange(-n3, n3).to(device).float()/n3
            [X,Y,Z]  = torch.meshgrid(xx,yy,zz)

            # set frequencies interval
            freqInt=torch.linspace(0,0.5, x1.shape[0]).to(device)

            if freqInt[-1] != 0.5:
               freqInt = torch.cat([freqInt,torch.Tensor([.5]).to(device)], 0)


            # set radius
            W = X**2+Y**2+Z**2;
            W=W#.view(-1).contiguous().float()

            y = torch.zeros(len(freqInt)-1,1).to(device);

            for i in range(  len(freqInt)-1):
                indx = (W>=(freqInt[i]**2)).float() *( W<(freqInt[i+1]**2)).float();
                indx=indx.float()
                fftx1realIndx = fftx1real*indx;
                fftx1imagIndx = fftx1imag*indx;
                fftx2realIndx = fftx2real*indx;
                fftx2imagIndx = fftx2imag*indx;

                correlationReal=fftx1realIndx*fftx2realIndx+fftx1imagIndx*fftx2imagIndx
                correlationImag=fftx1imagIndx*fftx2realIndx- fftx1realIndx*fftx2imagIndx
                correlation=(correlationReal.sum()**2+correlationImag.sum()**2).sqrt()

                normx1=(fftx1realIndx**2+fftx1imagIndx**2).sum().sqrt()
                normx2=(fftx2realIndx**2+fftx2imagIndx**2).sum().sqrt()
                y[i] = correlation/(normx1*normx2)
                print("Correlation at " +str(resolution/(freqInt[i].cpu().item()+1e-12))+' is ' +str(y[i].item()))


            return freqInt[:-1].cpu()/resolution, y.cpu()





        
        
        
        
        
        
    def Create(self,epochs ):
        
        GroundTruthPath=self.GroundTruthPath()
        ReconstructionPath=self.ReconstructionPath()
        FigurePath=self.FigurePath()
        ReconstructedSequence=self.ReconstructedSequence( ReconstructionPath)
        UnstableVolume='Reconstruction'
        
        
        
        
        
        
        with open(FigurePath + '/config.cfg', 'w') as fp:
            self.args.config.write(fp)
        with open(FigurePath + '/config.txt', 'w') as fp:
            self.args.config.write(fp)
            
        GroundTruth=mrcfile.open(os.path.join(GroundTruthPath, 'GroundTruthFlipped.mrc')).data
        Reconstruction=mrcfile.open(os.path.join(ReconstructionPath, 'volumeRiboSynResampled.mrc')).data
        VolumeMask=self.gen.G.mask.cpu().numpy()  
        
        
        size=GroundTruth.shape[0]
        #freq,fscBefore=self.fsc(torch.Tensor(GroundTruth).cuda(), torch.Tensor(Reconstruction).cuda(), torch.Tensor(VolumeMask).cuda(), size )
        resolutionAchieved= 1#    resolutionAchieved2=1/freq[(fscBefore>0.143).sum()].item()
        #print("Resolution previous ", resolutionAchieved)
        
        ReconstructionRotated=np.transpose(np.flip(Reconstruction, 0),(2,1,0))
        ReconstructionRotated=   ReconstructionRotated.reshape(-1).reshape(GroundTruth.shape)
            
        #ReconstructionRegistered=self.registerVolumes(GroundTruth, ReconstructionRotated, VolumeMask, FigurePath, StableVolume='GroundTruth')
        ReconstructionRegistered=ReconstructionRotated
        with mrcfile.new(os.path.join(FigurePath, 'registered'+UnstableVolume+'.mrc' ), overwrite=True) as m  :
                        m.set_data(ReconstructionRegistered)  
        #with mrcfile.new(os.path.join(FigurePath, 'GroundTruth.mrc' ), overwrite=True) as m  :
        #                m.set_data(GroundTruth)  
        '''
        for flipdim in range(3):  
            for i in itertools.permutations([0,1,2]):

                print("flip axis"+str(flipdim)+"Trying permutation" +str(i))

                ReconstructionRotated=np.transpose(np.flip(Reconstruction, flipdim),i)
                ReconstructionRotated=np.sum(ReconstructionRotated*GroundTruth)/np.sum(ReconstructionRotated*ReconstructionRotated)*ReconstructionRotated
                with mrcfile.new(os.path.join(FigurePath, 'registered'+UnstableVolume+str(i)+str(flipdim)+'.mrc' ), overwrite=True) as m  :
                        m.set_data(ReconstructionRotated)  
                
                ReconstructionRegistered=ReconstructionRotated#self.registerVolumes(GroundTruth, ReconstructionRotated, VolumeMask, FigurePath, StableVolume='GroundTruth')

                #freq,fscAfter=self.fsc(torch.Tensor(GroundTruth).cuda(), torch.Tensor(ReconstructionRotated).cuda(), torch.Tensor(VolumeMask).cuda(), size )
                resolutionAchievedNew= 20*np.log10(np.linalg.norm(GroundTruth)/np.linalg.norm(GroundTruth-ReconstructionRotated))#    resolutionAchieved2=1/freq[(fscAfter>0.5).sum()].item()

                print("Resolution for this permutation", resolutionAchievedNew)
                if resolutionAchieved>resolutionAchievedNew:
                    resolutionAchieved=resolutionAchievedNew
                    ReconstructionRegisteredNew=ReconstructionRegistered
                    print("permutation best till now--> flip axis"+str(flipdim)+"Trying permutation" +str(i))
                   
                    with mrcfile.new(os.path.join(FigurePath, 'registered'+UnstableVolume+'.mrc' ), overwrite=True) as m  :
                        m.set_data(ReconstructionRegisteredNew)  

        
                           
        ReconstructionRegistered=ReconstructionRegisteredNew
        
        ReconstructionRegistered =ReconstructionRotated  
        '''                       
        GroundTruth=torch.Tensor(GroundTruth).to(self.device)
        ReconstructionRegistered=torch.Tensor(ReconstructionRegistered).to(self.device)
        Reconstruction=torch.Tensor(Reconstruction).to(self.device)
        VolumeMask=torch.Tensor(VolumeMask).to(self.device)
        
        if 'Ribo' in self.GroundTruthPath():
            resolution=1.34
        elif 'Betagal' in self.GroundTruthPath():
             resolution=0.637
        elif 'RiboNew' in self.GroundTruthPath():
            resolution=1.77
        # plot fsc
        size=GroundTruth.shape[0]
        resolution=0.637
        freq,fscBefore=self.fsc(GroundTruth, Reconstruction, VolumeMask, size, resolution)
        freq, fscAfter=self.fsc(GroundTruth, ReconstructionRegistered, VolumeMask,size, resolution)
       
        
        resolutionAchieved1=1/freq[(fscAfter>0.143).sum()-1].item()
        resolutionAchieved2=1/freq[(fscAfter>0.143).sum()].item()
        resolutionAchieved3=1/freq[(fscAfter>0.143).sum()+1].item()
        print("Resolution achieved " ,resolutionAchieved1, resolutionAchieved2, resolutionAchieved3)
        
        fig=plt.figure(1)
        
        plt.plot(freq.numpy(), fscBefore.numpy(), '-k', freq.numpy(),fscAfter.numpy(), '-b', freq.numpy(), 0.5*np.ones((len(freq.numpy()),1)), '--k')
       
        fig.savefig(os.path.join(FigurePath, 'FSC.png'))
        
       
        # plot Projections
        self.gen.X.data= GroundTruth
        real_samples, realNoisy, realCTF, realClean = self.gen(noise=None, depth=None, alpha=None, skipDownsampling=True, ChangeAngles=False)
        
        self.gen.X.data= ReconstructionRegistered
        fake_samples, fakeNoisy, fakeCTF, fakeClean = self.gen(noise=None, depth=None, alpha=None, skipDownsampling=True, ChangeAngles=False)
       
        
        name='Real Data'
        save_fig_single_separate( real_samples.cpu().data, FigurePath, name , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        
        name='Fake Data'
        save_fig_single_separate( fake_samples.cpu().data, FigurePath, name , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        
        name='Real Clean Data'
        save_fig_single_separate( realClean.cpu().data, FigurePath, name, nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        
        name='Fake Clean Data'
        save_fig_single_separate( fakeClean.cpu().data, FigurePath, name, nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        
        name='Real CTF Data'
        save_fig_single_separate( realCTF.cpu().data, FigurePath, name, nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        
        name='Fake CTF Data'
        save_fig_single_separate( fakeCTF.cpu().data, FigurePath, name, nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        
        
        
        
        sequence_samples=[]
        sequenceNoisy=[]
        sequenceCTF=[]
        sequenceClean=[]
        
        for i, volume in enumerate(ReconstructedSequence):
             baseName=str(100*float(i+1)/len(ReconstructedSequence)).zfill(2)+' Iteration ' 
             Reconstruction=mrcfile.open( volume).data
            
             ReconstructionRotated=np.transpose(np.flip(Reconstruction, 0),(2,1,0))
             #ReconstructionRotated=np.sum(ReconstructionRotated*GroundTruth.cpu().numpy())/np.sum(ReconstructionRotated*ReconstructionRotated)*ReconstructionRotated
             ReconstructionRotated=   ReconstructionRotated.reshape(-1).reshape(292,292,292)
                
                
             with mrcfile.new(os.path.join(FigurePath, 'registered'+UnstableVolume+baseName+'.mrc' ), overwrite=True) as m  :
                        m.set_data(ReconstructionRotated)  
            
             self.gen.X.data= torch.Tensor(ReconstructionRotated).to(self.device)
                
             proj_samples, projNoisy, projCTF, projClean = self.gen(noise=None, depth=None, alpha=None, skipDownsampling=True, ChangeAngles=False)
             sequence_samples.append(proj_samples)
             sequenceNoisy.append(projNoisy)
             sequenceCTF.append(projCTF)
             sequenceClean.append(projClean)
            
             
             
            
             name= baseName + 'Fake Clean Data'
             save_fig_single_separate( projClean.cpu().data, FigurePath, name , nrow=3, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
            
             '''
             name= baseName + 'Fake Data'
             save_fig_single_separate( proj_samples.cpu().data, FigurePath, name , nrow=3, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
            
            
             name= baseName + 'Fake CTF Data'
             save_fig_single_separate( projCTF.cpu().data, FigurePath, name , nrow=3, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
             '''   
             
             
             
            
            
        
    
       