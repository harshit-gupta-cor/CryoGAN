import os
import time
import copy
import numpy as np
import torch as th
import astra
import torch
from torch import Tensor
import sys
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
from torchvision import models
import dataSet_Clean as dataSet
from IPython.core.debugger import set_trace
import IPython.core.debugger
from copy import  copy, deepcopy
from config import Config as cfg
plt.switch_backend('agg')
import kornia
class ProGAN:



    def __init__(self, args=None, filename=None):
        args.BATCH_SIZE=args.batch_size
        self.args=args
        
        self.plot = libs.plot.Plotter()
        # Create the Generator and the Discriminator
        self.gen = cryoGenerator( args=self.args).to(self.args.device[0])
        
        if self.args.AlgoType != "generate":
            
    
            self.dis_device=self.args.device[-1]
  
            self.dis=DownsamplingCNN(args).to(self.dis_device)
            self.dis.apply(lambda m: weights_init(m, self.args))

            if self.args.UseOtherDis:
                
                listdir = os.listdir('./Results/'+self.args.dataset)
                listdir= ['./Results/'+self.args.dataset+'/'+dirname for dirname in listdir if self.args.name[:-5] in dirname and self.args.data_type not in dirname]
                if len(listdir)>0:
                    listdir.sort(key=lambda x: os.path.getctime(x))
                    dispath=listdir[-1]+'/Discriminator.pth'
                    print("Loading discriminator from "+dispath)
                    self.dis.load_state_dict(torch.load(dispath))
                else:
                    print("Couldnt find a previous saved dis")
        
           
            print('gen: ', self.gen)
            print('dis: ', self.dis)

            self.gen.G.scalar=torch.nn.Parameter(self.gen.G.scalar)
            
   
        
    def train(self, dataset, num_workers=3, feedback_factor=100,
              log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=1): 

        from DataTools import get_data_loader
        print("Starting the training process ... ")   
        data = get_data_loader(dataset,  self.args.batch_size, num_workers=0)    
        
        total_batches = len(iter(data))
        epoch_gen_iterations=total_batches//(self.args.dis_iterations+1)
        total_gen_iterations=epoch_gen_iterations*self.args.epochs 
        
        self.epoch_gen_iterations=epoch_gen_iterations
        self.total_gen_iterations=total_gen_iterations
        
        self.gen.train()
        self.dis.train() 

        self.initOptimizer()
        self.initScheduler(epoch_gen_iterations)
        
        
        OUTPUT_PATH=self.output_path()
        
        torch.save(self.dis.state_dict(), OUTPUT_PATH+'/Discriminator.pth')
          
        step=-1
        global_time = time.time()
        
            
        print("starting Training")
        
         
        for epoch in range(1, self.args.epochs+ 1):
                    
            self.epoch=epoch          
            start =timer()  # record time at the start of epoch
            
            
            if step>-1:
                
                dataset = dataSet.Cryo(args=self.args)
                data = get_data_loader(dataset, self.args.batch_size, num_workers=0) 
                
            if self.args.GaussianFilterProjection:
  
                print("Gaussian Blurring everything with sigma " +str(self.args.GaussianSigma ))
                self.args.GaussianSigmaEpochInitial=self.args.GaussianSigma
                self.args.GaussianSigmaEpochFinal=self.args.GaussianSigma*self.args.GaussianSigmaGamma 
       
            dataIter=iter(data)  
   
            
            for i in range(epoch_gen_iterations):  
                if self.args.GaussianFilterProjection:
                    perc=(float(i)/(epoch_gen_iterations-1) )
                    self.args.GaussianSigma=(perc *self.args.GaussianSigmaEpochFinal+  (1.0-perc)*self.args.GaussianSigmaEpochInitial) 
                
                step=step+1
                self.step=step
                self.ratio=float(step)/(total_gen_iterations)

                self.gen.VolumeMaskRadius=self.args.VolumeMaskSize                          
               
                    
                dis_loss=0
                wass_loss=0

                
                for Diter in range(self.args.dis_iterations):
                   
                    images=dataIter.next()
                    
                    dis_lossIter, wass_lossIter, real_samples = self.optimize_discriminator(images)
                    dis_loss=dis_loss + dis_lossIter/float(self.args.dis_iterations)
                    wass_loss=wass_loss + wass_lossIter/float(self.args.dis_iterations)


                gen_loss, fake_samples, projNoisy, projCTF, projClean, Noise = self.optimize_generator(images, multipleNoise=True)
               
                with torch.no_grad():
                    
                    self.gen.Constraint(ratio=self.ratio)


                
                self.plot.tick()
                self.plot.plot('DLoss', dis_loss)
                self.plot.plot('GLoss', gen_loss)
                self.plot.plot('WLoss', wass_loss)

               
                if gen_loss != gen_loss:
                    print("nan detected!!")
                    sys.exit()

                PercentageEpochDone=100.0*float(i)/epoch_gen_iterations
                PercentageRecDone=100.0*float(step)/total_gen_iterations
                
                if (i%feedback_factor) == 0:
                    elapsed = time.time() - global_time
                    elapsed = str(timedelta(seconds=elapsed)).split(".")[0]
                    

                    print("Elapsed: [%s]  Epoch: %d PercEpoch: %f PercRecDone: %f  wass_loss : %f"
                          % (elapsed, epoch, PercentageEpochDone, PercentageRecDone, wass_loss ))
                    
                    print("Scalar1: [%f] Scalar2: [%f] "%( torch.exp(self.gen.G.scalar[0]).item(), torch.exp(self.gen.G.scalar[1]).item() ))
                   
                    # also write the losses to the log file:
                    if not os.path.exists(log_dir):   
                        os.makedirs(log_dir)
                    log_file = os.path.join(log_dir, "loss.log")
                    with open(log_file, "a") as log:
                        log.write(str(step) + "\t" + str(dis_loss) +
                                  "\t" + str(gen_loss) + "\n")

                    with th.no_grad():
                        iteration=str(epoch).zfill(3) + "_" +str(i).zfill(4)

                        save_fig_double(real_samples.cpu().data,fake_samples.cpu().data, 
                                        OUTPUT_PATH, "Proj", iteration=iteration, 
                                        Title1='Real', Title2='Fake_' + str(iteration), 
                                        doCurrent=True, sameColorbar=False)

                        save_fig_double(projClean.cpu().data,projCTF.cpu().data,
                                        OUTPUT_PATH, "ProjClean", iteration=iteration,
                                        Title1='ProjClean', Title2='ProjCTF' + str(iteration),
                                        doCurrent=True, sameColorbar=False)
                        if self.args.NumItersToSkipProjection<step:
                            save_fig_double(self.projClean_grad.cpu().data,self.projCTF_grad.cpu().data,
                                            OUTPUT_PATH, "ProjGrad", iteration=iteration,
                                            Title1='ProjCleanGrad', Title2='ProjCTFGrad' + str(iteration),
                                            doCurrent=True, sameColorbar=False)
     
                        self.SaveVolume(OUTPUT_PATH)

                   #===================    

                        self.plot.save_plots(OUTPUT_PATH)
                        
                      
                        torch.save(self.gen.G.scalar,OUTPUT_PATH+"/scalar")
                        
                self.schedulerD.step()
                self.schedulerG.step()
                self.schedulerScalar.step()
             
            
            self.SaveVolume(OUTPUT_PATH, 'reconstruction_Epoch_'+str(int(epoch)))
            torch.save(self.gen.G.scalar,OUTPUT_PATH+'/scalar_reconstruction_Epoch_'+str(int(epoch)))

            
            if (10*(epoch))%self.args.epochs==0:
                    self.SaveVolume(OUTPUT_PATH, 'reconstruction_'+str(int(PercentageRecDone+1)))
                    torch.save(self.gen.G.scalar,OUTPUT_PATH+'/scalar_reconstruction_'+str(int(PercentageRecDone+1)))
                    
            stop = timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

        self.gen.eval()
        self.dis.eval()
        print("Training completed ...")

    
    
    def initOptimizer(self):     
         # define the optimizers for the discriminator and generator
        
        if self.args.symmetryNormalizedLR:
            n= self.args.SymmetryN if self.args.SymmetryType != 'none' else 1
            n= 2*self.args.SymmetryN if self.args.SymmetryType == 'D' else n
        else:
            n=1
            

        
        self.gen_optim = torch.optim.Adam([{'params' : self.gen.X}], 
                                              lr=self.args.gen_lr/n, 
                                              betas=(self.args.gen_beta_1, self.args.gen_beta_2), 
                                              eps=self.args.gen_eps, 
                                              weight_decay=self.args.gen_weight_decay)
    
        dictionary=list(self.dis.parameters())
        
        self.dis_optim = torch.optim.Adam(dictionary, 
                                              lr=self.args.dis_lr, 
                                              betas=(self.args.dis_beta_1, self.args.dis_beta_2), 
                                              eps=self.args.dis_eps, 
                                              weight_decay=self.args.dis_weight_decay)
    
        dictionary=[  {"params" :self.gen.G.scalar}]
      
        self.scalar_optim=torch.optim.Adam(dictionary , 
                                                   lr=self.args.scalar_lr, 
                                                   betas=(self.args.scalar_beta_1, self.args.scalar_beta_2),                                  
                                                   eps=self.args.scalar_eps,
                                                   weight_decay=self.args.scalar_weight_decay)
        

    def initScheduler(self, epoch_GIterations):
        step_size=self.args.step_size
        gamma=self.args.gamma
        print("Lr decreases every "+ str(step_size)+ " epochs")

        self.schedulerD = optim.lr_scheduler.StepLR(self.dis_optim, 
                                                    step_size=int(epoch_GIterations*step_size),
                                                    gamma=gamma)
        self.schedulerG = optim.lr_scheduler.StepLR(self.gen_optim, 
                                                    step_size=int(epoch_GIterations*step_size), 
                                                    gamma=gamma)
      
        self.schedulerScalar=optim.lr_scheduler.StepLR(self.scalar_optim, 
                                                           step_size=int(epoch_GIterations*step_size), 
                                                           gamma=gamma)
   
    


    def optimize_discriminator(self, real_samples):

        """
        performs one step of weight update on discriminator using the batch of data
        :param real_batch: real samples batch
        :return: current loss (Wasserstein loss)
        """
        loss_val = 0       
        # generate a batch of samples
        with torch.no_grad():
            
            fake_samples,_,_,_,_ = self.gen(real_samples , GaussianSigma=self.args.GaussianSigma, ratio=self.ratio)
            fake_samples=fake_samples.detach()   
        
        real_samples=real_samples.to(self.dis_device)
        fake_samples=fake_samples.to(self.dis_device)
        
        loss, wass_loss = dis_loss(self.dis, real_samples, fake_samples, self.args.lambdaPenalty, self.args.lambda_drift,self.args.gamma_gradient_penalty )  
  
            
        if self.args.dis_clip_grad ==True:
            
            clip_norm_value=np.min([1e3+(self.args.dis_clip_norm_value-1e3)*0.5*self.step/float(self.epoch_gen_iterations), self.args.dis_clip_norm_value])
            
            dictionary=list(self.dis.parameters())
            torch.nn.utils.clip_grad_norm_(dictionary, clip_norm_value)
                
       
        loss.backward()      
        self.dis_optim.step()
        self.dis_optim.zero_grad()
        loss_val += loss.item()
        return loss_val, wass_loss, real_samples


    def optimize_generator(self, real_samples, multipleNoise):

        """
        performs one step of weight update on generator for the given batch_size
        :param real_batch: batch of real samples
        :return: current loss (Wasserstein estimate)
        """
       
        # generate fake samples:
    

        fake_samples, projNoisy, projCTF, projClean,Noise = self.gen(real_samples, GaussianSigma=self.args.GaussianSigma, ratio=self.ratio, multipleNoise=multipleNoise)
        
        projClean.retain_grad()
        projCTF.retain_grad()

        
        fake_samples=fake_samples.to(self.dis_device)
        
        
        loss= gen_loss(self.dis, fake_samples)

        # optimize the generator
        
        loss.backward()
        self.projClean_grad=projClean.grad
        self.projCTF_grad=projCTF.grad
      
  
        if self.args.gen_clip_grad ==True:    
    
            clip_norm_value=np.min([1+(self.args.gen_clip_norm_value-1)*0.5*self.step/float(self.epoch_gen_iterations), self.args.gen_clip_norm_value])
          
            torch.nn.utils.clip_grad_value_(self.gen.X, clip_norm_value) 
        
    
        self.gen_optim.step()
        self.gen_optim.zero_grad()

        
        if  self.args.LearnSigma:
            if self.args.scalar_clip_grad:
                torch.nn.utils.clip_grad_value_(self.gen.G.scalar, self.args.scalar_clip_norm_value)  
            self.scalar_optim.step()
            self.scalar_optim.zero_grad()
            
        
        return loss.item(), fake_samples, projNoisy, projCTF, projClean, Noise
    

    def SaveVolume(self, sample_dir, name=None):

        with torch.no_grad():   
                volume=self.gen.Volume
        
        for i, volumeSingle in enumerate(volume):
            
                         if name is not None:

                            nameSingle=name+"_"+str(i+1)+'.mrc'
                         
                         else:
                            nameSingle='volume_'+str(i+1)+'.mrc'
                    
                         with mrcfile.new(os.path.join(sample_dir, nameSingle), overwrite=True) as m:

                            m.set_data(volumeSingle.data.cpu().numpy())



                     
    def SaveVolumeSlices(self, OUTPUT_PATH, iteration)  :  
        
     
        volume=self.gen.Volume
        symmetryAxis=None
        
        if self.args.SymmetryType !='none':
            symmetryAxis=np.argmax(self.gen.X.shape)
            volume=torch.transpose(volume, int(1), int(symmetryAxis-1) )
        
        i= np.random.randint(0, volume.shape[0])
        volumeSingle =volume[i]
            
        save_fig(volumeSingle.unsqueeze(1).cpu().data, 
                OUTPUT_PATH, 'volume_'+str(i+1), iteration=iteration, 
                 Title='volume' + str(iteration), doCurrent = True)
        

                           
    
    def output_path(self):
        OUTPUT_PATH= './'+'Results/'
        if os.path.exists(OUTPUT_PATH)==False:
            os.mkdir(OUTPUT_PATH)
        OUTPUT_PATH = OUTPUT_PATH+self.args.dataset+'/' # output path where result (.e.g drawing images, cost, chart) will be stored
        if os.path.exists(OUTPUT_PATH)==False:
            os.mkdir(OUTPUT_PATH)

        # create an output directory with appropriate name
        date = datetime.now().strftime("%Y-%m-%dT%H-%M")
        if self.args.name is not None:
                name = self.args.name + '-'
        else:
                name =''
        OUTPUT_PATH= os.path.join(OUTPUT_PATH, name + date)

        if os.path.exists(OUTPUT_PATH) ==False :
            os.mkdir(OUTPUT_PATH)
        # save the arguments to a file 
        with open(OUTPUT_PATH + '/config.cfg', 'w') as fp:
            self.args.config.write(fp)
        with open(OUTPUT_PATH + '/config.txt', 'w') as fp:
            self.args.config.write(fp)
            
        return OUTPUT_PATH
    
    def SaveOrthoSlices(self, volume, FigurePath, name):
        N=volume.shape[-1]
        slice_yz=volume[N//2,:,:].squeeze().unsqueeze(0).unsqueeze(0)
        slice_xz=volume[:,N//2,:].squeeze().unsqueeze(0).unsqueeze(0)
        slice_xy=volume[:,:,N//2].squeeze().unsqueeze(0).unsqueeze(0)
        
        save_fig_single_separate(slice_yz, FigurePath, name+'-yz' , nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        save_fig_single_separate( slice_xz, FigurePath, name +'-xz', nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
        save_fig_single_separate( slice_xy, FigurePath, name +'-xy', nrow=1, iteration=None, doCurrent=False, figshow=False, Title=None, scaleEach=False)
    

            
  