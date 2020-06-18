""" Loads program configuration into a config object."""

from backports import configparser 
import numpy as np
from copy import copy, deepcopy
class Config:
    def __init__(cls, filename):
        cls.load_config(filename)
 
    def load_config(cls, filename):
        config = configparser.SafeConfigParser()
        config.read(filename)
        cls.config = config

        s = "general"
        cls.name = config.get(s, 'name')
        cls.AlgoType = config.get(s, 'AlgoType')
        if cls.AlgoType == 'generate':
            cls.DatasetSize = config.getint(s, 'DatasetSize')
            cls.snr_ratio=config.getfloat(s, 'snr_ratio')
            
        cls.DownSampleRate=config.getint(s, 'DownSampleRate')

        cls.use2gpu = config.getboolean(s, 'use2gpu')
        cls.device = []
        try:
            cls.NoisePerCTF = config.getint(s, 'NoisePerCTF')
        except:
            cls.NoisePerCTF=1
       
        
        s = "data"
        cls.dataset = config.get(s, 'dataset')
        cls.dataset_name = config.get(s, 'dataset_name')
        cls.data_type = config.get(s, 'data_type')
        cls.InstantRealData = config.getboolean(s, 'instantrealdata')
        try:
            cls.UseOtherDis=config.getboolean(s,'useotherdis')
            cls.ThresholdResolution=config.getfloat(s, 'thresholdresolution')
            cls.PixelSize=config.getfloat(s, 'pixelsize')  
            cls.FreezeResolution=config.getboolean(s,'freezeresolution')
            cls.ReferencePath=config.get(s,'referencepath')
            
        
        except:
            pass
 
    
        s = "generator"

        
        cls.use_deep_prior = config.getboolean(s, 'use_deep_prior')
        cls.deep_prior_net = config.get(s, 'deep_prior_net')
        cls.deep_prior_net_inC = config.getint(s, 'deep_prior_net_inC')
        cls.init_dp_net = config.getboolean(s, 'init_dp_net')
        cls.init_dp_lr = config.getfloat(s, 'init_dp_lr')
        cls.init_dp_lr_step = config.getint(s, 'init_dp_lr_step')
        cls.init_dp_niter = config.getint(s, 'init_dp_niter')
        cls.init_path = config.get(s, 'init_path')
        
        cls.FourierProjector=config.getboolean(s, 'FourierProjector')
        cls.VolumeDomain=config.get(s, 'VolumeDomain').lower()

        cls.VolumeSize = config.getint(s, 'VolumeSize')
        cls.VolumeNumbers = config.getint(s, 'VolumeNumbers')

        cls.RawProjectionSize = config.getint(s, 'RawProjectionSize')
        cls.ProjectionStep = config.getfloat(s, 'ProjectionStep')
       

        cls.AngleDistribution = config.get(s, 'AngleDistribution').lower()
        if cls.AngleDistribution not in ['uniform', 'cylinder','cylindernoisy']:
            raise Exception( "unrecognized AngleDistribution: " + cls.AngleDistribution )
    
        cls.UseVolumeGenerator=config.getboolean(s, 'UseVolumeGenerator')
        cls.SymmetryType = config.get(s, 'SymmetryType')
        cls.SymmetryN = config.getint(s, 'SymmetryN')
        
        cls.CTF = config.getboolean(s, 'CTF')
        cls.ChangeCTFs = config.getboolean(s, 'ChangeCTFs')
        cls.CTFSize = config.getint(s, 'CTFSize')
        cls.valueAtNyquist=config.getfloat(s, 'valueAtNyquist')
        cls.skipNoise=config.getboolean(s,'skipNoise')
        
        cls.sigma = config.getfloat(s, 'sigma')
        cls.LearnSigma = config.getboolean(s, 'LearnSigma')
        cls.NumItersToSkipProjection = config.getint(s, 'NumItersToSkipProjection')
      
        cls.sigma1 = config.getfloat(s, 'sigma1')
        cls.sigma2 = config.getfloat(s, 'sigma2')
        cls.scalar = config.getfloat(s, 'scalar')
        cls.ContrastVector=config.getboolean(s, 'contrastvector')
        cls.dc = config.getfloat(s, 'dc')

        cls.Translation = config.getboolean(s, 'Translation')
        cls.EstimateFirstMomentTranslation = config.getboolean(s, 'EstimateFirstMomentTranslation')
        cls.TranslationVariance = config.getfloat(s, 'TranslationVariance')
        cls.NormalizeProjections = config.getboolean(s, 'NormalizeProjections')
        try:
            cls.ProjScale = config.getfloat(s, 'ProjScale')
        except:
            pass
        cls.InvertProjections = config.getboolean(s, 'InvertProjections')
       

        cls.Constraint = config.get(s, 'Constraint')
        cls.ValueConstraint =config.getboolean(s, 'ValueConstraint')
        cls.Value =config.getfloat(s, 'Value')

        cls.UseEstimatedAngles=config.getboolean(s, 'UseEstimatedAngles')
        cls.UseEstimatedTranslations=config.getboolean(s, 'UseEstimatedTranslations')
        cls.UseEstimatedDefocuses=config.getboolean(s, 'UseEstimatedDefocuses')
        cls.UseEstimatedNoise=config.getboolean(s, 'UseEstimatedNoise')
        cls.NoisePoisson=config.getboolean(s, 'noisepoisson')
        cls.NoiseUnet=config.getboolean(s, 'noiseunet')
        
        s = "masks"
        
        
        cls.VolumeMask = config.getboolean(s, 'VolumeMask')
        cls.VolumeMaskSize = config.getfloat(s, 'VolumeMaskSize')
        
        cls.ProjectionMask = config.getboolean(s, 'ProjectionMask')
        cls.ProjectionMaskSize = config.getfloat(s, 'ProjectionMaskSize')
        
        cls.ProjectionMaskFourier = config.getboolean(s, 'ProjectionMaskFourier')
        cls.ProjectionMaskFourierSize = config.getfloat(s, 'projectionmaskfouriersize')
        
        cls.ResolutionMask=config.getboolean(s, 'resolutionmask')
        cls.ResolutionMaskLimit=config.getfloat(s, 'resolutionmasklimit')
        
        cls.weinerFilter=config.getboolean(s, 'weinerfilter')
        cls.weinerConstant=config.getfloat(s, 'weinerconstant')
        
            
        

        
        s = "postprocessing"
     
        try:
      
            cls.kernel_size = config.getint(s, 'KernelSize')
            cls.AveragingStepSize = config.getfloat(s, 'AveragingStepSize')
        except:
            pass

        s = "discriminator"
        cls.Dtype = config.getint(s, 'Dtype')
        cls.FourierDiscriminator = config.getboolean(s, 'FourierDiscriminator')
        cls.leak_value = config.getfloat(s, 'leak_value')
        cls.Lambda = config.getfloat(s, 'Lambda')
        cls.bias = config.getboolean(s, 'bias')            
        cls.num_channel_Discriminator=config.getint(s, 'num_channel_Discriminator')
        cls.num_layer_Discriminator=config.getint(s, 'num_layer_Discriminator')
        cls.num_N_Discriminator=config.getint(s, 'num_n_discriminator')

        s = "optimization"
        cls.epochs=config.getint(s, 'epochs')
        cls.batch_size = config.getint(s, 'batch_size')
        cls.lambdaPenalty=config.getfloat(s, 'lambdaPenalty')
        try:
            cls.lambdaPenaltyGamma=config.getfloat(s, 'lambdaPenaltyGamma')
        except:
            cls.lambdaPenaltyGamma=1
        cls.step_size = config.getfloat(s, 'step_size')
        cls.gamma = config.getfloat(s, 'gamma')
        cls.AveragingGradientIteration=config.getint(s, 'AveragingGradientIteration')
        cls.symmetryNormalizedLR=config.getboolean(s, 'symmetrynormalizedlr')
        cls.ConnectedComponent=config.getboolean(s, 'connectedcomponent')
        cls.GaussianFilterProjection=config.getboolean(s, 'gaussianfilterprojection')
        cls.GaussianSigma=config.getfloat(s, 'gaussiansigma')
        cls.GaussianSigmaGamma=config.getfloat(s, 'gaussiansigmagamma')
        s = "optimization_gen"
        cls.gen_optimizer = config.get(s, 'gen_optimizer')
        cls.gen_lr = config.getfloat(s, 'gen_lr')
        cls.gen_momentum = config.getfloat(s, 'gen_momentum')
        cls.gen_beta_1=config.getfloat(s, 'gen_beta_1')
        cls.gen_beta_2=config.getfloat(s, 'gen_beta_2')
        cls.gen_eps=config.getfloat(s, 'gen_eps')
        cls.gen_clip_grad=config.getboolean(s, 'gen_clip_grad')
        cls.gen_clip_norm_value=config.getfloat(s, 'gen_clip_norm_value')
        cls.gen_weight_decay=config.getfloat(s, 'gen_weight_decay')

        s = "optimization_dis"
        cls.dis_iterations = config.getint(s, 'dis_iterations')
        cls.dis_optimizer = config.get(s, 'dis_optimizer')
        cls.dis_lr = config.getfloat(s, 'dis_lr')
        cls.dis_beta_1=config.getfloat(s, 'dis_beta_1')
        cls.dis_beta_2=config.getfloat(s, 'dis_beta_2')
        cls.dis_eps=config.getfloat(s, 'dis_eps')
        cls.dis_clip_grad=config.getboolean(s, 'dis_clip_grad')
        cls.dis_clip_norm_value=config.getfloat(s, 'dis_clip_norm_value')
        cls.dis_weight_decay=config.getfloat(s, 'dis_weight_decay')

        s = "optimization_scalar"

        cls.scalar_optimizer = config.get(s, 'scalar_optimizer')
        cls.scalar_lr = config.getfloat(s, 'scalar_lr')
        cls.scalar_beta_1=config.getfloat(s, 'scalar_beta_1')
        cls.scalar_beta_2=config.getfloat(s, 'scalar_beta_2')
        cls.scalar_eps=config.getfloat(s, 'scalar_eps')      
        cls.scalar_weight_decay=config.getfloat(s, 'scalar_weight_decay')
        cls.scalar_clip_grad=config.getboolean(s, 'scalar_clip_grad')
        cls.scalar_clip_norm_value=config.getfloat(s, 'scalar_clip_norm_value')




        s = "display"
        cls.showActivation = config.getboolean(s, 'showActivation')


         
 
    def calc_derived_params(cls):
        cls.RawProjectionSize = int(cls.RawProjectionSize // cls.DownSampleRate)
        cls.ProjectionSize= cls.RawProjectionSize
        cls.VolumeSize = int(cls.VolumeSize // cls.DownSampleRate)
        cls.num_layer_Discriminator=cls.num_layer_Discriminator-int(np.log2(cls.DownSampleRate))
        
        cls.CTFSize = (cls.CTFSize//2)*2+1 # ensure odd
        

        #if cls.AlgoType == 'generate':
        #    cls.batch_size = 1 # needed for CTF generation
        cls.BATCH_SIZE= cls.batch_size
    
#     def __copy__(self):
#         cls = self.__class__
#         result = cls.__new__(cls)
#         result.__dict__.update(self.__dict__)
#         return result

#     def __deepcopy__(self, memo):
#         cls = self.__class__
#         result = cls.__new__(cls)
#         memo[id(self)] = result
#         for k, v in self.__dict__.items():
#             setattr(result, k, deepcopy(v, memo))
#         return result
