# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 01:45:00 2018

@author: 
    wuyifan
    mayiping
"""

import os
import numpy as np
import time
import sys
from PIL import Image
import scipy.misc
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    def __init__ (self, pathModel, transCrop):
       
        model = models.densenet201(pretrained = True)

        classes = 1 # here we can modify the classes of data
        model.classifier = nn.Linear(1920, classes)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('best_BCE.dense.tar')               
       
        model.load_state_dict(checkpoint['state_dict'])
       
        self.model = model.module.features
        self.model.eval()
        
        # Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
       # transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    

    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        # Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
       
        output = self.model(input.cuda())
        # Generate heatmap
        heatmap = np.zeros((7, 7, 1), dtype=np.float64)
        weight = self.weights.data.cpu().numpy()
        len = weight.shape[0]
        output = output.data.cpu().numpy()
        for i in range(len):
            heatmap[:, :, 0] += output[0, i]*weight[i]
        
        heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())*255
        heatmap = np.uint8(heatmap)
        heatmap = cv2.resize(heatmap, (transCrop, transCrop))
        
        '''
        print('weights shape: ',self.weights.shape)
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
      
        print('heatmap size =', heatmap.shape)
        npHeatmap = heatmap.cpu().data.numpy()
        '''
        
        imgOriginal = cv2.imread(pathImageFile)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)      
        img = heatmap * 0.5 + imgOriginal * 0.5            
        cv2.imwrite(pathOutputFile, img)
        
        
def GetHeatMap(path_input, path_output, path_model):
    
    pathInputImage = path_input
    pathOutputImage = path_output
    pathModel = path_model

    transCrop = 224

    h = HeatmapGenerator(pathModel, transCrop)
    h.generate(pathInputImage, pathOutputImage, transCrop)

