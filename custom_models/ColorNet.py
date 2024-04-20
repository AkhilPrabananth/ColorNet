import torch
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from fastai.torch_basics import *
from fastai.callback.hook import *
from torch import nn


class ColorNet(Module):
    def __init__(self,*layers):
        super().__init__()
        self.LSTM_bool=False
        self.layers = nn.ModuleList(layers)
        print(layers[3])

    def init_lstm(self,size=(1024)):
        self.LSTM_bool=True
        self.lstm=NDIM_LSTM(size)
        print("LSTM initiated")
    
    def forward(self,x):
        res=x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            res.orig, nres.orig = None, None
            res = nres
            
        return res
