""" Loads program configuration into a config object."""

from backports import configparser 
import numpy as np
from copy import copy, deepcopy
from IPython.core.debugger import set_trace
class Config(object):
    def __init__(self, filename):
        self.load_config(filename)
        self.calc_derived_params()
 
    def load_config(self, filename):
        config = configparser.SafeConfigParser()
        config.read(filename)
        self.config = config

        s = "general"
        self.name = config.get(s, 'name')
        self.AlgoType = config.get(s, 'AlgoType')
        if self.AlgoType=='generate':
            self.snr_ratio=config.getfloat(s, 'snr_ratio')
        self.DatasetSize = config.getint(s, 'DatasetSize')
        
            
        self.DownSampleRate=config.getint(s, 'DownSampleRate')

        self.use2gpu = config.getboolean(s, 'use2gpu')
        self.device = []
        try:
            self.NoisePerCTF = config.getint(s, 'NoisePerCTF')
        except:
            self.NoisePerCTF=1
        try:   
            self.PhaseFlipping=config.getboolean(s, 'PhaseFlipping')
        except:
            self.PhaseFlipping=False
        s = "data"
        self.dataset = config.get(s, 'dataset')
        self.dataset_name = config.get(s, 'dataset_name')
        self.data_type = config.get(s, 'data_type')
        self.InstantRealData = config.getboolean(s, 'instantrealdata')
        try:
            self.UseOtherDis=config.getboolean(s,'useotherdis')
            self.ThresholdResolution=config.getfloat(s, 'thresholdresolution')
            self.PixelSize=config.getfloat(s, 'pixelsize')  
            self.FreezeResolution=config.getboolean(s,'freezeresolution')
            self.ReferencePath=config.get(s,'referencepath')
            
        
        except:
            pass
 
    
        s = "generator"

        
        self.use_deep_prior = config.getboolean(s, 'use_deep_prior')
        self.deep_prior_net = config.get(s, 'deep_prior_net')
        self.deep_prior_net_inC = config.getint(s, 'deep_prior_net_inC')
        self.init_dp_net = config.getboolean(s, 'init_dp_net')
        self.init_dp_lr = config.getfloat(s, 'init_dp_lr')
        self.init_dp_lr_step = config.getint(s, 'init_dp_lr_step')
        self.init_dp_niter = config.getint(s, 'init_dp_niter')
        self.init_path = config.get(s, 'init_path')
        
        self.FourierProjector=config.getboolean(s, 'FourierProjector')
        self.VolumeDomain=config.get(s, 'VolumeDomain').lower()

        self.VolumeSize = config.getint(s, 'VolumeSize')
        self.VolumeNumbers = config.getint(s, 'VolumeNumbers')

        self.RawProjectionSize = config.getint(s, 'RawProjectionSize')
        self.ProjectionStep = config.getfloat(s, 'ProjectionStep')
       

        self.AngleDistribution = config.get(s, 'AngleDistribution').lower()
        if self.AngleDistribution not in ['uniform', 'cylinder','cylindernoisy']:
            raise Exception( "unrecognized AngleDistribution: " + self.AngleDistribution )
    
        self.UseVolumeGenerator=config.getboolean(s, 'UseVolumeGenerator')
        self.SymmetryType = config.get(s, 'SymmetryType')
        self.SymmetryN = config.getint(s, 'SymmetryN')
        
        self.CTF = config.getboolean(s, 'CTF')
        self.ChangeCTFs = config.getboolean(s, 'ChangeCTFs')
        self.CTFSize = config.getint(s, 'CTFSize')
        self.valueAtNyquist=config.getfloat(s, 'valueAtNyquist')
        self.skipNoise=config.getboolean(s,'skipNoise')
        
        self.sigma = config.getfloat(s, 'sigma')
        self.LearnSigma = config.getboolean(s, 'LearnSigma')
        self.NumItersToSkipProjection = config.getint(s, 'NumItersToSkipProjection')
      
        self.sigma1 = config.getfloat(s, 'sigma1')
        self.sigma2 = config.getfloat(s, 'sigma2')
        self.scalar = config.getfloat(s, 'scalar')
        self.ContrastVector=config.getboolean(s, 'contrastvector')
        self.dc = config.getfloat(s, 'dc')

        self.Translation = config.getboolean(s, 'Translation')
        self.EstimateFirstMomentTranslation = config.getboolean(s, 'EstimateFirstMomentTranslation')
        self.TranslationVariance = config.getfloat(s, 'TranslationVariance')
        self.NormalizeProjections = config.getboolean(s, 'NormalizeProjections')
        try:
            self.ProjScale = config.getfloat(s, 'ProjScale')
        except:
            pass
        self.InvertProjections = config.getboolean(s, 'InvertProjections')
       

        self.Constraint = config.get(s, 'Constraint')
        self.ValueConstraint =config.getboolean(s, 'ValueConstraint')
        self.Value =config.getfloat(s, 'Value')

        self.UseEstimatedAngles=config.getboolean(s, 'UseEstimatedAngles')
        self.UseEstimatedTranslations=config.getboolean(s, 'UseEstimatedTranslations')
        self.UseEstimatedDefocuses=config.getboolean(s, 'UseEstimatedDefocuses')
        self.UseEstimatedNoise=config.getboolean(s, 'UseEstimatedNoise')
        self.NoisePoisson=config.getboolean(s, 'noisepoisson')
        self.NoiseUnet=config.getboolean(s, 'noiseunet')
        
        s = "masks"
        
        
        self.VolumeMask = config.getboolean(s, 'VolumeMask')
        self.VolumeMaskSize = config.getfloat(s, 'VolumeMaskSize')
        
        self.ProjectionMask = config.getboolean(s, 'ProjectionMask')
        self.ProjectionMaskSize = config.getfloat(s, 'ProjectionMaskSize')
        
        self.ProjectionMaskFourier = config.getboolean(s, 'ProjectionMaskFourier')
        self.ProjectionMaskFourierSize = config.getfloat(s, 'projectionmaskfouriersize')
        
        self.ResolutionMask=config.getboolean(s, 'resolutionmask')
        self.ResolutionMaskLimit=config.getfloat(s, 'resolutionmasklimit')
        
        self.weinerFilter=config.getboolean(s, 'weinerfilter')
        self.weinerConstant=config.getfloat(s, 'weinerconstant')
        
            
        

        
        s = "postprocessing"
     
        try:
      
            self.kernel_size = config.getint(s, 'KernelSize')
            self.AveragingStepSize = config.getfloat(s, 'AveragingStepSize')
        except:
            pass

        s = "discriminator"
        self.Dtype = config.getint(s, 'Dtype')
        self.FourierDiscriminator = config.getboolean(s, 'FourierDiscriminator')
        self.leak_value = config.getfloat(s, 'leak_value')
        self.Lambda = config.getfloat(s, 'Lambda')
        self.bias = config.getboolean(s, 'bias')            
        self.num_channel_Discriminator=config.getint(s, 'num_channel_Discriminator')
        self.num_layer_Discriminator=config.getint(s, 'num_layer_Discriminator')
        self.num_N_Discriminator=config.getint(s, 'num_n_discriminator')

        s = "optimization"
        self.epochs=config.getint(s, 'epochs')
        self.batch_size = config.getint(s, 'batch_size')
        self.lambdaPenalty=config.getfloat(s, 'lambdaPenalty')
        try:
            
            self.lambda_drift=config.getfloat(s, 'lambda_drift')
            self.gamma_gradient_penalty=config.getfloat(s,'gamma_gradient_penalty')
            self.lambdaPenaltyGamma=config.getfloat(s, 'lambdaPenaltyGamma')
        except:
            self.lambdaPenaltyGamma=1
            self.lambda_drift=0
            self.gamma_gradient_penalty=1
            
            
        self.step_size = config.getfloat(s, 'step_size')
        self.gamma = config.getfloat(s, 'gamma')
        self.AveragingGradientIteration=config.getint(s, 'AveragingGradientIteration')
        self.symmetryNormalizedLR=config.getboolean(s, 'symmetrynormalizedlr')
        self.ConnectedComponent=config.getboolean(s, 'connectedcomponent')
        self.GaussianFilterProjection=config.getboolean(s, 'gaussianfilterprojection')
        self.GaussianSigma=config.getfloat(s, 'gaussiansigma')
        self.GaussianSigmaGamma=config.getfloat(s, 'gaussiansigmagamma')
        
        s = "optimization_gen"
        self.gen_optimizer = config.get(s, 'gen_optimizer')
        self.gen_lr = config.getfloat(s, 'gen_lr')
        self.gen_momentum = config.getfloat(s, 'gen_momentum')
        self.gen_beta_1=config.getfloat(s, 'gen_beta_1')
        self.gen_beta_2=config.getfloat(s, 'gen_beta_2')
        self.gen_eps=config.getfloat(s, 'gen_eps')
        self.gen_clip_grad=config.getboolean(s, 'gen_clip_grad')
        self.gen_clip_norm_value=config.getfloat(s, 'gen_clip_norm_value')
        self.gen_weight_decay=config.getfloat(s, 'gen_weight_decay')

        s = "optimization_dis"
        self.dis_iterations = config.getint(s, 'dis_iterations')
        self.dis_optimizer = config.get(s, 'dis_optimizer')
        self.dis_lr = config.getfloat(s, 'dis_lr')
        self.dis_beta_1=config.getfloat(s, 'dis_beta_1')
        self.dis_beta_2=config.getfloat(s, 'dis_beta_2')
        self.dis_eps=config.getfloat(s, 'dis_eps')
        self.dis_clip_grad=config.getboolean(s, 'dis_clip_grad')
        self.dis_clip_norm_value=config.getfloat(s, 'dis_clip_norm_value')
        self.dis_weight_decay=config.getfloat(s, 'dis_weight_decay')

        s = "optimization_scalar"

        self.scalar_optimizer = config.get(s, 'scalar_optimizer')
        self.scalar_lr = config.getfloat(s, 'scalar_lr')
        self.scalar_beta_1=config.getfloat(s, 'scalar_beta_1')
        self.scalar_beta_2=config.getfloat(s, 'scalar_beta_2')
        self.scalar_eps=config.getfloat(s, 'scalar_eps')      
        self.scalar_weight_decay=config.getfloat(s, 'scalar_weight_decay')
        self.scalar_clip_grad=config.getboolean(s, 'scalar_clip_grad')
        self.scalar_clip_norm_value=config.getfloat(s, 'scalar_clip_norm_value')




        s = "display"
        self.showActivation = config.getboolean(s, 'showActivation')


         
 
    def calc_derived_params(self):
        self.RawProjectionSize = int(self.RawProjectionSize // self.DownSampleRate)
        self.ProjectionSize= self.RawProjectionSize
        self.VolumeSize = int(self.VolumeSize // self.DownSampleRate)
        self.num_layer_Discriminator=self.num_layer_Discriminator-int(np.log2(self.DownSampleRate))
        
        self.CTFSize = (self.CTFSize//2)*2+1 # ensure odd
        

        #if self.AlgoType == 'generate':
        #    self.batch_size = 1 # needed for CTF generation
        self.BATCH_SIZE= self.batch_size
    
#     def __copy__(self):
#         self = self.__class__
#         result = self.__new__(self)
#         result.__dict__.update(self.__dict__)
#         return result

#     def __deepcopy__(self, memo):
#         set_trace()
#         self = self.__class__
#         result = self.__new__(self)
#         memo[id(self)] = result
#         for k, v in self.__dict__.items():
#             setattr(result, k, deepcopy(v, memo))
#         return result
