import sys
from PyQt5  import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize 

import torch as th
import torch 
import torchvision as tv
import CryoGAN_Clean as pg
import dataSet_Clean as dataSet
import argparse
from config import Config as cfg
from IPython.core.debugger import set_trace



def initDataset(args):
       
        dataset = dataSet.Cryo(args=args)
        return dataset




def Central():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default="Configs/train_New_Betagal-Synthetic-NoiseCTF.cfg", help="Specify config file", metavar="FILE")
    args = parser.parse_args()
    
    cfg.load_config(args.config)
    cfg.calc_derived_params()

    # select the device to be used for training
    cfg.device = [th.device('cpu'), th.device('cpu')]
    if th.cuda.is_available():
        if cfg.use2gpu: 
            cfg.device = [th.device('cuda:0'), th.device('cuda:1')]
        else:
            cfg.device = [th.device('cuda'), th.device('cuda')]
    
    # some parameters:

    dataset=initDataset(cfg)


    pro_gan = pg.ProGAN(args=cfg)
  

    pro_gan.train(dataset=dataset, num_workers=1)

    
       
   
    

class MyWindow(QMainWindow):
    def __init__(self, dictConfig):
        super(MyWindow, self).__init__()
        self.setGeometry(200,200,1000,1000)
        self.setWindowTitle("CryoGAN")
        self.dictConfig=dictConfig
        self.initUI()
        
    def initUI(self):
        
        
        self.b1=QPushButton(self)
        self.b1.setText("Click Me")
        self.b1.clicked.connect(self.clicked)
        
        self.variable=[]
        self.val=[]
        self.sectionlabel=[]
        position=0
        for section in self.dictConfig:
            position=position+25
            
            self.sectionlabel.append(QLabel(self))
            self.sectionlabel[-1].setText(section[0])
            self.sectionlabel[-1].move(20, position)
            self.sectionlabel[-1].resize(150, 20)
            
            for i,variablename in enumerate(section[1]):
                position=position+25
                self.inputBox(variablename, [40, position, 150, 20] )
        
        
    def inputBox(self, name, position):
        self.variable.append( QLabel(self) )
        self.variable[-1].setText(name)
        self.val.append(QLineEdit(self))

        self.val[-1].move(position[0]+150, position[1])
        self.val[-1].resize(position[2], position[3])
        self.variable[-1].move(position[0], position[1])
        self.variable[-1].resize(position[2], position[3])
    
    def clicked(self):
        #Central()
        print([(name.text(), val.text()) for name, val in zip(self.variable,self.val) ])
        #self.label.setText("you pressed the button")
        #self.update()
        
    def update(self):
        self.label.adjustSize()
    
    
def window(dictConfig):
        app=QApplication(sys.argv)
        app.setStyle('Fusion')
        win=MyWindow(dictConfig)
        win.show()
        sys.exit(app.exec_())

dictConfig=[]
dictConfig.append(["general",["name", "AlgoType", "DatasetSize", "snr_ratio", "DownSampleRate", "use2gpu", "device", "NoisePerCTF"]] )
                       
dictConfig.append(["data",["dataset", "dataset_name", "data_type", "UseOtherDis","ThresholdResolution", "PixelSize", "FreezeResolution" ]] )

 


window(dictConfig)




