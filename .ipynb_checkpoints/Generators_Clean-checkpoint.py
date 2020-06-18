from __future__ import print_function
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch import Tensor
import astra
import numpy as np
import Functions
from Functions.Functions import *
from Functions.FunctionsCTF import *
from Functions.FunctionsHandle import *
from Functions.FunctionsSymmetry import *
from Functions.FunctionsSaveImage import *
from Functions.FunctionsFourier import *
from Functions.FunctionsGenerator import *
import Networks_Clean

from torch.utils.data.dataset import Dataset
from  torch.utils.data.dataloader import DataLoader
from VarphiGenerator_Clean import *
import mrcfile
import scipy
#%%


class ProjectorModule(nn.Module):
    def __init__(self, args):
        
        super(ProjectorModule,self).__init__()
        self.args=args
        self.X, self.Xd, self.S, self.UnS=InitVolume(args=self.args) 

        self.args.iteration=0  

        self.scalar=self.args.scalar*torch.ones(2).cuda()
        
        self.B= torch.nn.InstanceNorm2d(num_features=1,momentum=0.0)
        self.ConvMode=False
        
        self.VarphiGenerator=VarphiGenerator(args)
        self.VarphiGeneratorLoader= iter(DataLoader(self.VarphiGenerator ,  batch_size=args.batch_size,
                                                    shuffle=True, num_workers=0,
                                                    drop_last=True, pin_memory=False))
        self.CurrentAngles=None
        self.snr=0
        self.sigmaNoiseGenerated=0
        self.snriteration=0
        
            

    def forward(self, X, ChangeAngles=True,  ratio=0):
    
        X=self.Sym(X, self.S)        
     
        angles, translation, defocusU, defocusV, astigmatism, Noise=self.VarphiGeneratorBatch()
        
        if ChangeAngles==True or self.CurrentAngles is None:
            self.CurrentAngles=angles
        
        X=torch.exp(self.scalar[0])*X 
        
        proj,projClean=self.Projection(X,  self.CurrentAngles)       
        
        proj= self.Translation(proj, translation)
            
        projCTF= self.CTF(proj, defocusU, defocusV, astigmatism)
        
        projCTF= self.TranslationCropping(projCTF, translation) 
       
        projNoisy= self.NoiseAddition(projCTF, proj, Noise)
        
        return projNoisy, projCTF, projClean, X.detach(), Noise
        
        
    def Sym(self, X, S):
        if self.args.SymmetryType != 'none':
            X = S(X) 
        return X
   

    def Translation(self, proj, translation):
        if self.args.Translation:
            
            translationx, translationy=translation.unbind(-1)
           
            proj = TranslationPadding(proj, translationx, translationy)
        return proj
    
    def TranslationCropping(self, projCTF, translation):
        
        if self.args.Translation:
            translationx, translationy=translation.unbind(-1)
            
            projCTF=TranslationCropping( projCTF, translationx, translationy, self.args)
            
        return projCTF
        
    def Projection(self, X, angles):    
        
        vectors=angles_to_vectors(angles, self.args.ProjectionStep)             
        proj= Projection(X[0], self.args.RawProjectionSize, vectors.cpu())                           
        projClean=proj
        return proj, projClean

    def CTF(self, proj, defocusU, defocusV, astigmatism):
            
        self.hFourier,self.hSpatial= CTFGenerator(self.args, defocusU, defocusV, astigmatism)
            
        projCTF = CTFforward(self.hSpatial, proj)

        return projCTF
            
    def NoiseAddition(self,projCTF,projClean, Noise, multipleNoise=False):
    
        if self.args.AlgoType != 'generate':
               
                    projNoisy=    projCTF+ torch.exp(self.scalar[1])*Noise[:,0,:,:].unsqueeze(1) 
                        
        else:
                ScaledNoise=torch.exp(self.scalar[1])*Noise
                
                energy1=projCTF.pow(2).view(projCTF.shape[0], -1).sum(1).sqrt()
                
                energy2=ScaledNoise.pow(2).view(projCTF.shape[0], -1).sum(1).sqrt()
                 
                sigma=(energy1/energy2).mean()/self.args.snr_ratio                 
                
                self.sigmaNoiseGenerated=(self.snriteration*self.sigmaNoiseGenerated+sigma)/(self.snriteration+1)
                
                projNoisy=   ( projCTF+self.sigmaNoiseGenerated*torch.exp(self.scalar[1])*Noise )
                    
                self.snriteration=self.snriteration+1

        if self.args.NormalizeProjections==True:
            
            projNoisy = self.B(projNoisy)
        
        if self.args.InvertProjections:
            projNoisy *= -1


        return projNoisy
    

    def VarphiGeneratorBatch(self):

            try:
                out = self.VarphiGeneratorLoader.next()
            except StopIteration: # reshuffle the dataset

                self.VarphiGeneratorLoader= iter(torch.utils.data.DataLoader(self.VarphiGenerator,batch_size=self.args.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)   )
                    
                out = self.VarphiGeneratorLoader.next()

            return  [element.to('cuda', non_blocking=True) for element in out]
        

