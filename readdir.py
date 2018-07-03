# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 01:45:00 2018

@author: huomingjia
"""



#importcv2
from PIL import Image
import numpy as np
import csv
import sys
from feature import densemodel
import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms

def get_model():
    model = models.densenet201(pretrained = True)
   
    classes = 2 # here we can modify the classes of data
    model.classifier = nn.Linear(1920, classes) 
    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load('model_retrieval.tar')
    model.load_state_dict(checkpoint['state_dict'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        #transforms.Resize(320),
        #transforms.CenterCrop(224),
        # optional: random rotatation before training or not
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30), 
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    return model, train_transforms

def csv_process(in_dir, out_dir, csv_name):
    f = open(out_dir, "w")
    model, transform = get_model()
    cnt = 0
    with open(in_dir+csv_name, 'r') as csvfile:
        img_reader = csv.reader(csvfile)
        for row in img_reader:
            if row[1] == '1':
                print(cnt)
                cnt+=1 
                f.write(row[0])
                f.write(" ")
                tmp = row[0].find("/")+1
                imgname = in_dir+row[0][tmp:]
                #print(model)            	
                img = Image.open(imgname).convert('RGB')
                img = transform(img)
                img_PIL = transforms.ToPILImage()(img).convert('RGB')
                model_feature = model.module.features               
                net = densemodel(model_feature, img_PIL)
                cntcnt=0
                #net.show()
                weight = net.extract_lastlayer()
                weight1 = weight[0].data.cpu().numpy()
                b = 0.0
                for i in range(1920):
                    b = 0.0
                    for j in range(7):
                        a = sum(weight1[i][j])    
                        b += a  
                    b /= 49  
                    f.write(str(b)+' ')
                #print(cntcnt)
                #return
                f.write('\r\n')
    with open(in_dir+'valid.csv', 'r') as csvfile:
        img_reader = csv.reader(csvfile)
        for row in img_reader:
            if row[1] == '1':
                print(cnt)
                cnt+=1 
                f.write(row[0])
                f.write(" ")
                tmp = row[0].find("/")+1
                imgname = in_dir+row[0][tmp:]
                #print(model)            	
                img = Image.open(imgname).convert('RGB')
                img = transform(img)
                img_PIL = transforms.ToPILImage()(img).convert('RGB')
                model_feature = model.module.features               
                net = densemodel(model_feature, img_PIL)
                cntcnt=0
                #net.show()
                weight = net.extract_lastlayer()
                weight1 = weight[0].data.cpu().numpy()
                b = 0.0
                for i in range(1920):
                    b = 0.0
                    for j in range(7):
                        a = sum(weight1[i][j])    
                        b += a  
                    b /= 49  
                    f.write(str(b)+' ')
                #print(cntcnt)
                #return
                f.write('\r\n')
    f.close()

