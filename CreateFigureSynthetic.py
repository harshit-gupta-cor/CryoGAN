

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
from Functions.FunctionsSaveImage import *
import mrcfile
import scipy.io 
import scipy
import libs.plot
# temp timing
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import itertools
plt.switch_backend('agg')
import dataSet_Clean as dataSet
from IPython.core.debugger import set_trace
 
 
        
        
class CreateFigure:
  

    def __init__(self, args):
      
   
        from torch.optim import Adam
        from torch.nn import DataParallel
        
        self.args=args
        self.args.BATCH_SIZE=1
        self.args.Batch_Size=1
        self.args.batch_size=1
        device="cuda:0"
        self.args.SymmetryType='none'
        
        if "Freezing-15A-Result-8A" in self.args.name:
            self.args.GaussianSigma=self.args.GaussianSigma*(self.args.GaussianSigmaGamma)**(12)
        
        else:
            self.args.GaussianSigma=self.args.GaussianSigma*(self.args.GaussianSigmaGamma)**(self.args.epochs)
            
        self.args.GaussianFilterProjection=False
        self.gen = cryoGenerator( args=self.args).to(device)
        self.device=device
       
   
        
    def FSC(self, Path1, Path2, OutPath, LP=False, Average=False):   
        if 'Ribo' in self.args.dataset:
            resolution=2*1.34
        elif 'Betagal' in self.args.dataset:
             resolution=self.args.DownSampleRate*0.637
             print("Pixel size is "+str(resolution))
        elif 'RiboNew' in self.args.dataset:
            resolution=1.77
        else:
            resolution=1.06
            
        command="focus.postprocess "+ Path1+" "+ Path2+" --angpix "+str(resolution)+" --automask --auto_bfac -1,-1 --out "+ OutPath 
        print(command)
        os.system(command)
        if not os.path.exists(OutPath+"_fsc-masked.txt"):
            time.sleep(5)
        
        fsc_masked=np.loadtxt(OutPath+"_fsc-masked.txt")
        fsc_masked_freq=np.loadtxt(OutPath+"_fsc-masked-freq.txt")
        fsc_unmasked=np.loadtxt(OutPath+"_fsc-unmasked.txt")
        fsc_unmasked_freq=np.loadtxt(OutPath+"_fsc-unmasked-freq.txt")
        
        FSC_masked_recon=self.ResolutionAtThreshold(fsc_masked_freq, fsc_masked)
        
        
        if LP:
            print("Low Pass")
            command="focus.postprocess "+ Path1+" "+ Path1+" --angpix "+str(resolution)+" --automask --auto_bfac -1,-1 "+"--lowpass "+("% 2.2f" % FSC_masked_recon).zfill(4)+" --out "+ OutPath+"_LP_Even"
            
            os.system(command)
            if not os.path.exists(OutPath+"_LP_Even"+"_fsc-masked.txt"):
                time.sleep(5)
            
            
            command="focus.postprocess "+ Path2+" "+ Path2+" --angpix "+str(resolution)+" --automask --auto_bfac -1,-1 "+"--lowpass "+("% 2.2f" % FSC_masked_recon).zfill(4)+" --out "+ OutPath+"_LP_Odd"
            
            os.system(command)
            if not os.path.exists(OutPath+"_LP_Odd"+"_fsc-masked.txt"):
                time.sleep(5)
                
            if Average:
                with mrcfile.new(OutPath+"_Average.mrc") as m:
                    half1= mrcfile.open(Path1).data
                    half2= mrcfile.open(Path2).data
                    m.set_data((half1+half2)/2.0)
                command="focus.postprocess "+ OutPath+"_Average.mrc"+" "+ OutPath+"_Average.mrc"+" --angpix "+str(resolution)+" --automask --auto_bfac -1,-1 "+"--lowpass "+("% 2.2f" % FSC_masked_recon).zfill(4)+" --out "+ OutPath+"_LP_Average"
                os.system(command)
                if not os.path.exists(OutPath+"_LP_Average"+"_fsc-masked.txt"):
                    time.sleep(5)
            
                

        return fsc_masked, fsc_unmasked, fsc_unmasked_freq
    
    
    def AlignBetagal(self, path, volume, mask =None):  
        
        normPrev=-np.Inf
        with mrcfile.open(path, permissive=True) as m:
            volumeGT=m.data
        if self.args.VolumeSize==170:
                volumeGT=volumeGT[5:-5, 5:-5, 5:-5]
        if self.args.VolumeSize>volumeGT.shape[-1]:
                    gap=self.args.VolumeSize-volumeGT.shape[-1]
                    volumeGT=np.pad(volumeGT, (gap//2,))
                    
        
        volumeGTFiltered=scipy.ndimage.gaussian_filter(volumeGT,  2)
        scaling=np.max(volumeGT)/np.max(volume)
        volume=scaling*volume
        volumeFiltered=scipy.ndimage.gaussian_filter(volume,  2)
        for flip in [True, False]:  
            for i in itertools.permutations([0,1,2]):
                 if flip:
                     volumeRotatedTempFiltered=np.flip(np.transpose(volumeFiltered,i), 0)
                     volumeRotatedTemp=np.flip(np.transpose(volume,i), 0)
                 else:
                     volumeRotatedTempFiltered=np.transpose(volumeFiltered,i)
                     volumeRotatedTemp=np.transpose(volume,i)
                
                 #normNew=np.sum((volumeRotatedTempFiltered)*(volumeGT))/ ( np.sum( (volumeRotatedTemp)**2)**0.5 * np.sum((volumeGT)**2)**0.5 )
                 normNew=-np.sum((volumeRotatedTempFiltered-volumeGTFiltered)**2)
                 if normNew>normPrev:
                     normPrev=normNew
                     volumeRotated=volumeRotatedTemp
        
        if mask is not None:    
            return mask*volumeRotated/scaling
        else:
            return volumeRotated/scaling
        
        
    def Create(self,epochs ):
        self.gen.G.BoundaryMask=int(self.args.VolumeMaskSize*self.args.VolumeSize)//2
        
        
           
        EvenPath=self.EvenPath()
     
        
        FigurePath=self.FigurePath()
        
        #######Saving Mask
        GroundTruthPath=self.GroundTruthPath()
        with mrcfile.open(GroundTruthPath, permissive=True) as m:
            volumeGT=m.data
          
            if self.args.VolumeSize<volumeGT.shape[-1]:
                volumeGT=volumeGT[5:-5, 5:-5, 5:-5]
            if self.args.VolumeSize>volumeGT.shape[-1]:
                    gap=self.args.VolumeSize-volumeGT.shape[-1]
                    volumeGT=np.pad(volumeGT, (gap//2,))
                    
            mask=np.array(binary_dilation((volumeGT>0.0001), iterations=10), dtype=np.float32)
        
        #mask=np.ones(mask.shape, dtype=np.float32)
        print("GT mask is one everywhere")
        with mrcfile.new(FigurePath+"/mask.mrc") as m:
            m.set_data(mask)
            
        
        
        
        ReconstructedSequenceEven=self.ReconstructedSequence( EvenPath)
        
        
        
        print("Even path is " +EvenPath)
    
        print("Figure path is " +FigurePath)
        
        from DataTools import get_data_loader
        dataset = dataSet.Cryo(args=self.args)
        data = get_data_loader(dataset, self.args.batch_size, num_workers=0) 
        dataIter=iter(data)  
        
        
        
        with open(FigurePath + '/config.cfg', 'w') as fp:
            self.args.config.write(fp)
        with open(FigurePath + '/config.txt', 'w') as fp:
            self.args.config.write(fp)
            
        #######Saving Half half
        Even=mrcfile.open(os.path.join(EvenPath, 'volume_1.mrc')).data
       
            

        
        Even=torch.Tensor(Even)
       
        
         #### Save ortho Slices###

     
       
        self.SaveOrthoSlices( Even, FigurePath, "ReconEven-Slice")
       
        
        
        Even=Even.to(self.device).unsqueeze(0)
        
        
        
        ####################################
            
       
        #scalarGT=torch.load(GroundTruthPath+'/scalar')
        scalarEven=torch.load(EvenPath+'scalar')
     
        

        # plot fsc
        size=Even.shape[0]
        
        for numFigure in range(1):
        
       
            # plot Projections
            #############GT
            

            real_samples=dataIter.next()


            name='Real Data '+str(numFigure+1)
            save_fig_single_separate( real_samples.cpu().data, FigurePath, name , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)

            



            ############Even
            self.gen.X.data= Even
            self.gen.G.scalar.data=scalarEven
            
            fake_samples, projNoisy, projCTF, projClean = self.gen(None, ChangeAngles=True)

            name='Fake Data Even '+str(numFigure+1)
            save_fig_single_separate( fake_samples.cpu().data, FigurePath, name , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)


            name='Fake Clean Data Even '+str(numFigure+1)
            save_fig_single_separate( projClean.cpu().data, FigurePath, name, nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)


            name='Fake CTF Data Even '+str(numFigure+1)
            save_fig_single_separate( projCTF.cpu().data, FigurePath, name, nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
            
            
            
            
            sequence_samples=[]
            sequenceNoisy=[]
            sequenceCTF=[]
            sequenceClean=[]

            for i, volume in enumerate(ReconstructedSequenceEven):
                
                 FigurePathCurrent=os.path.join(FigurePath,str(10*(i+1)).zfill(2)+'_Iteration', 'Even/')
                 baseName=str(10*(i+1)).zfill(2)+'_Iteration_Even' 
                 Reconstruction=mrcfile.open( volume).data
                 
                 Reconstruction=self.AlignBetagal(GroundTruthPath, Reconstruction, mask=mask)
                 with mrcfile.new(FigurePathCurrent+ baseName+".mrc", overwrite=True) as m:
                    m.set_data(Reconstruction)
            
                 Reconstruction =torch.Tensor(Reconstruction)
                 self.SaveOrthoSlices( Reconstruction, FigurePathCurrent, baseName)


                 self.gen.X.data= Reconstruction.to(self.device).unsqueeze(0)
  
                    
                 self.gen.G.scalar.data=scalarEven

                 proj_samples, projNoisy, projCTF, projClean = self.gen(None, ChangeAngles=False)
                 sequence_samples.append(proj_samples)
                 sequenceNoisy.append(projNoisy)
                 sequenceCTF.append(projCTF)
                 sequenceClean.append(projClean)




                 name= baseName + 'Fake Clean Data '+str(numFigure+1)
                 save_fig_single_separate( projClean.cpu().data, FigurePathCurrent, name , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)

                 
                 name= baseName + 'Fake Data'+str(numFigure+1)
                 save_fig_single_separate( proj_samples.cpu().data, FigurePathCurrent, name , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)


                 name= baseName + 'Fake CTF Data'+str(numFigure+1)
                 save_fig_single_separate( projCTF.cpu().data, FigurePathCurrent, name , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
                


         
        
        fsc_names=[]
        fsc_masked=[]
        fsc_unmasked=[]
        FSC_number_masked=[]
        FSC_number_unmasked=[]
        
        fsc_names_even=[]
        fsc_masked_even=[]
        fsc_unmasked_even=[]        
        FSC_number_masked_even=[]
        FSC_number_unmasked_even=[]
        
    

            
        
    
        
        
        #######################
        for i in range(len(ReconstructedSequenceEven)):
            FigurePathCurrentEven=os.path.join(FigurePath,str(10*(i+1)).zfill(2)+'_Iteration', 'Even/')
            FigurePathCurrent=os.path.join(FigurePath,str(10*(i+1)).zfill(2)+'_Iteration/')
            
            
            baseName=str(10*(i+1)).zfill(2)+'_Iteration' 
            fsc_masked_even_single, fsc_unmasked_even_single, freq=self.FSC(FigurePathCurrentEven+baseName+"_Even.mrc",GroundTruthPath, os.path.join(FigurePathCurrentEven,"FSC_GT_Even_"+baseName))
            
            fsc_names_even.append(baseName.replace("_", "-"))
            fsc_masked_even.append(fsc_masked_even_single)
            fsc_unmasked_even.append(fsc_unmasked_even_single)
            
            FSC_masked_even_single=self.ResolutionAtThreshold(freq, fsc_masked_even_single)
            FSC_unmasked_even_single=self.ResolutionAtThreshold(freq, fsc_unmasked_even_single)
        
            FSC_number_masked_even.append(FSC_masked_even_single)
            FSC_number_unmasked_even.append(FSC_unmasked_even_single)
        
        
        self.Figure(fsc_masked_even, freq, fsc_names_even, FSC_number_masked_even, os.path.join(FigurePath,"FSC_Masked_GT_Even"))
        self.Figure(fsc_unmasked_even, freq, fsc_names_even, FSC_number_unmasked_even, os.path.join(FigurePath,"FSC_Unmasked_GT_Even"))
        
     
        
        ########################
        
#         for files in os.listdir(os.path.join(FigurePath,"*.png")):
#             os.remove(files)
       
        for i in range(10):
            FigurePathCurrentEven=os.path.join(FigurePath,str(10*(i+1)).zfill(2)+'_Iteration', 'Even/')
            FigurePathCurrent=os.path.join(FigurePath,str(10*(i+1)).zfill(2)+'_Iteration/')
        
#             os.system("rm  "+FigurePathCurrent+"*mask.*")
#             os.system("rm  "+FigurePathCurrent+"*unmasked*")
            
            
#             os.system("rm  "+FigurePathCurrentEven+"*mask.*")
#             os.system("rm  "+FigurePathCurrentEven+"*unmasked*")
            
            
             
             
    def GroundTruthPath(self):
        OUTPUT_PATH= './'+'Datasets/'

        OUTPUT_PATH = OUTPUT_PATH+self.args.dataset       
        name = self.args.dataset_name if self.args.dataset_name is not None else ''        
        OUTPUT_PATH= os.path.join(OUTPUT_PATH, name ) 
        
        
        #OUTPUT_PATH='./Datasets/Betagal-Synthetic/GroundTruth_0.mrc'
        #OUTPUT_PATH='./Datasets/Betagal-Synthetic/molmap2.5-Downsampled'+str(self.args.DownSampleRate)+'-translationOff-EstimatedNoise-snr_ratio_0.4/GroundTruth_0.mrc'
        OUTPUT_PATH=OUTPUT_PATH+'/GroundTruth_0.mrc'
        return OUTPUT_PATH
    
    def EvenPath(self):
        OUTPUT_PATH= './'+'Results/'
        OUTPUT_PATH = OUTPUT_PATH+self.args.dataset+'/'
        name = self.args.name  
        print(name)
        lists=[ OUTPUT_PATH+folderName for folderName in os.listdir(OUTPUT_PATH) if name in folderName]
       
        lists.sort(key=lambda x: os.path.getctime(x))
        
        
        print(lists)
        OUTPUT_PATH=lists[-1]
        OUTPUT_PATH=OUTPUT_PATH+'/'
       


        return OUTPUT_PATH
  
     
        
    def ReconstructedSequence(self, OUTPUT_PATH):
        
        lists=[ os.path.join(OUTPUT_PATH,fileName) for fileName in os.listdir(OUTPUT_PATH) if ('.mrc' in fileName) and ('reconstruction'  in fileName) ]
      
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
        
        for i in range(10):
            name=str(10*(i+1)).zfill(2)+'_Iteration'
            os.mkdir(os.path.join(OUTPUT_PATH,name ) )
            os.mkdir(os.path.join(OUTPUT_PATH,name, 'Even' ) )
            
            
        return OUTPUT_PATH 
    
    
        
     
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
    def SaveOrthoSlices(self, volume, FigurePath, name):
        N=volume.shape[-1]
        slice_yz=volume[N//2,:,:].squeeze().unsqueeze(0).unsqueeze(0)
        slice_xz=volume[:,N//2,:].squeeze().unsqueeze(0).unsqueeze(0)
        slice_xy=volume[:,:,N//2].squeeze().unsqueeze(0).unsqueeze(0)
        
        save_fig_single_separate(slice_yz, FigurePath, name+'-yz' , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        save_fig_single_separate( slice_xz, FigurePath, name +'-xz', nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        save_fig_single_separate( slice_xy, FigurePath, name +'-xy', nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        
    def Figure(self, fsc, freq, fsc_names, FSC_numbers, figure_name, ext=".png", TimePer=40):
                plt.figure()

              
                for i,fsc_single in enumerate(fsc):
                    plt.plot(freq, fsc_single, linewidth=1*(i==len(fsc)-1)+0.4)
                
                fsc_names=[str(TimePer*(i+1))+" mins: "+("% 2.2f" % FSC_number).zfill(4) +" A" for i,(fsc_name,FSC_number) in enumerate(zip(fsc_names,FSC_numbers) )]
                
                plt.legend(fsc_names)

                plt.title('Fourier Shell Correlation')
                plt.ylabel('FSC')
                plt.xlabel(r'Spatial frequency (1/$\rm \AA$)')
                plt.minorticks_on()
                ax = plt.gca()
                ax.set_yticks([0.143], minor=True)
                ax.set_yticklabels(["0.143"], minor=True)
#                 if options.refine_res_lim != None:
#                     ax.axvline(1.0 / options.refine_res_lim,
#                                linestyle='dashed', linewidth=0.75, color='m')
                plt.grid(b=True, which='both', linewidth=0.1)
                plt.savefig(figure_name + ext,
                            dpi=300)
                plt.close()
            
                plt.figure()
                
                fsc_numbers=[fsc_number for (i, fsc_number) in enumerate(FSC_numbers) if "Recon" not in fsc_names[i]]
                iterations=10*np.arange(1, len(fsc_numbers)+1)
                plt.plot(iterations, fsc_numbers)


                plt.title('Fourier Shell Correlation - Time')
                plt.ylabel('FSC')
                plt.xlabel(r'Percentage Iteration')
                plt.minorticks_on()
                plt.grid(b=True, which='both')
                plt.savefig(figure_name + "_Chronology"+ ext,
                            dpi=300)
                plt.close()
            
            
    def ResolutionAtThreshold(self,freq, fsc, thr=0.143, interp=True):
    
        if np.isscalar(thr):

            thr *= np.ones(fsc.shape)

        # i = 0
        for i, f in enumerate(fsc):

            # if f < thr and i > 0:
            if f < thr[i]:

                break

            # i += 1

        if i < len(fsc) - 1 and i > 1:

            if interp:

                y1 = fsc[i]
                y0 = fsc[i - 1]
                x1 = freq[i]
                x0 = freq[i - 1]

                delta = (y1 - y0) / (x1 - x0)

                res_freq = x0 + (thr[i - 1] - y0) / delta

            else:

                # Just return the highest resolution bin at which FSC is still higher than threshold:
                res_freq = freq[i - 1]

        elif i == 0:

            res_freq = freq[i]

        else:

            res_freq = freq[-1]

            

        return 1 / res_freq
    
    
    





        
                    
            
        
    
       