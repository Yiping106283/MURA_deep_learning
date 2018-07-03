# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:00:51 2018
@author: 
    wuyifan 
    mayiping
"""


import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from torch.optim.lr_scheduler import ReduceLROnPlateau

from os.path import exists, isdir, isfile, join
from os import getcwd
import shutil
from tqdm import tqdm
import numpy

from dataloadertrain import MuraDataset # here load module dataloader
from aucmeter import AUCMeter 

# here we define all the global var

def TrainModel(datadir,pathtomodel, pathsavemodel, workers,epochs,batch_size,lr,weight_decay):
    model = models.densenet201(pretrained = True)
   
    classes = 1 # here we can modify the classes of data
    model.classifier = nn.Linear(1920, classes) 
    model = torch.nn.DataParallel(model).cuda()
    #checkpoint = torch.load('model_best_BCE_batch8-1.dense.tar')
    checkpoint = torch.load(pathtomodel)
    model.load_state_dict(checkpoint['state_dict'])
    last_iter = checkpoint['best_roc_auc']
    print('last_iter = ', last_iter)
    
    # data loading
    data_dir =  datadir #'../mura/muraproc/' # data_dir_name is to be set
    train_dir = join(data_dir, 'train')
    train_csv = join(data_dir, 'train.csv')
    validate_dir = join(data_dir, 'valid')
    validate_csv = join(data_dir, 'valid.csv')
    test_dir = join(data_dir, 'test') # here remained for test
      
    # ensure that data loading is successful
    assert isdir(data_dir) and isdir(train_dir) and isdir(validate_dir) and isdir(test_dir)
    assert exists(train_csv) and isfile(train_csv) and exists(validate_csv) and isfile(validate_csv)

    # before feeding images into network, we normalize each image to same var and mean
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # scale image to 224*224, optional: random rotation
    train_transforms = transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
        # optional: random rotatation before training or not
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # call module MuraDataset, save it in train_data
    train_data = MuraDataset(train_csv, transform = train_transforms)

    # initial weights
    temp_weights = train_data.balanced_weights
    weights = torch.DoubleTensor(temp_weights)

    # here has several options: 
    # more optionals see at: https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/
    # sampler = torch.utils.data.sampler.SequentialSampler(train_data)
    # sampler = torch.utils.data.sampler.RandomSampler(train_data)  
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # define train_loader
    train_loader = data.DataLoader(
        train_data,
        batch_size = batch_size, # here we can adjust batch size
        # shuffle=True,  # here we can choose whether to shuffle
        num_workers = workers, # Load the data in parallel using multiprocessing workers
        sampler = sampler,
        pin_memory = True) # pin_memory is just for faster data transfers


    # define validate loader
    validate_loader = data.DataLoader(
        MuraDataset(validate_csv,
                    transforms.Compose([
#                        transforms.Resize(256),
#                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])),
        batch_size = 1, # here we can adjust batch size
        shuffle = False, # here we can choose whether to shuffle
        num_workers = workers, # Load the data in parallel using multiprocessing workers.
        pin_memory=True) # pin_memory is just for faster data transfers

    
    
    # define critical things for training 

    criterion = nn.BCELoss()
    # the second parameter in optim is learning rate
    # we can adjust the optim to SGD, etc.
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    best_validate_loss = 0
    best_roc_auc = 0
    # begin training
    
    for epoch in range(1, epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        validate_loss,roc_auc = validate(validate_loader, model, criterion, epoch)
        scheduler.step(validate_loss)
        # remember best accuracy and save checkpoint
        is_best = roc_auc > best_roc_auc
        best_roc_auc = max(roc_auc, best_roc_auc)
        best_validate_loss = max(validate_loss, best_validate_loss)
        save_checkpoint(
           { 'epoch': epoch + 1,
            'arch': 'densenet',
            'state_dict': model.state_dict(),
            'best_validate_loss': best_validate_loss,
            'best_roc_auc': best_roc_auc,
         }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    # this can be modified, sometimes we set it to zero
    train_accuracy = AverageMeter()
    train_losses = AverageMeter()

    # pbar is progress bar, tqdm is a module for showing progress bar
    pbar = tqdm(train_loader)
   # print('reach here: just after pbar = tqdm(train_loader)')
    
    # train process begin
    
    for i, (images, target, meta) in enumerate(pbar):
       # print('come into for loop')
        target = target.cuda() # moving data to gpu
       # print(type(target))
        # turing image and target to torchvariable
        image_var = Variable(images)
        label_var = Variable(target)
      
        # pass this batch through our model and get y_pred
        y_prediction = model(image_var)
        
        y_pred = torch.sigmoid(y_prediction)
        label_v = label_var.float()
        loss = criterion(y_pred, label_v.view(-1, 1))
        
        train_losses.update(loss.data[0], images.size(0))

        # update accuracy metric
        #print(y_prediction.data.shape, target.shape)
        _, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        train_accuracy.update(prec1[0], images.size(0))
        #print('prec1[0] = ', prec1[0])
        
        # compute gradient and do SGD
        optimizer.zero_grad() # here we can change the initial grad
        loss.backward()
        optimizer.step()

         # showing the progress bar
        pbar.set_description("EPOCH[{0}][{1}/{2}]".format(epoch, i, len(train_loader)))
        pbar.set_postfix(
            train_accuracy ="{acc.value: .4f} ({acc.average: .4f})".format(acc=train_accuracy),
            loss="{loss.value: .4f} ({loss.average: .4f})".format(loss=train_losses))
        
    return

def validate(validate_loader, model, criterion, epoch):
    # switch to evaluation mode
    model.eval()

    validate_accuracy = AverageMeter()
    validate_losses = AverageMeter()
    meta_data = []

    # show the progress bar
    pbar = tqdm(validate_loader)
    last_iter_patient = torch.tensor([1])
    last_iter_study = torch.tensor([1])
    image_var = None
    label_var = None
    
    for i, (images, target, meta) in enumerate(pbar):
        target = target.cuda() #move it to gpu
#        image_var = Variable(images, volatile=True)
#        label_var = Variable(target, volatile=True)
        '''   
        batch_s = target.data.cpu().numpy().shape[0]
        '''
        cur_patient = meta['patient']
        cur_study = meta['study']
        if i==0:
            image_var = Variable(images)
            label_var = Variable(target)
            last_iter_patient = cur_patient
            last_iter_study = cur_study
        elif (cur_study.numpy() == last_iter_study.numpy()).all() and (cur_patient.numpy() == last_iter_patient.numpy()).all():
            next_image_var = Variable(images)
            image_var = torch.cat((image_var, next_image_var), 0)
            next_label_var = Variable(target)
            label_var = torch.cat((label_var, next_label_var), 0)
        else:
            y_prediction = model(image_var)
            label_v = label_var.float()
            y_pred = torch.sigmoid(y_prediction)
            #print("y_pred", y_pred, "\n")
            y_pred = y_pred.mean()
            y_pred = y_pred.expand(1,1)
            #print("y_pred", y_pred, "\n")
            label_v = Variable(torch.tensor(label_v[0]))
            label_v = label_v.expand(1,1)
            #print("label", label_v, "\n")
            loss = criterion(y_pred, label_v.view(-1, 1))
            label_v = label_v.long()
            validate_losses.update(loss.data[0], images.size(0))
          #  if meta['img_filename'][0]=='../mura/muraproc/valid/XR_WRIST/patient11283/study2_negative/image1.png':
          #      print("!!!test!!y_pred.data target", y_pred, y_pred.data, target)
            # update accuracy metric on the GPU
            _, prec1 = accuracy(y_pred.data, label_v.data, topk=(1, 1))
            validate_accuracy.update(prec1[0], images.size(0))

            #print("prec1 = ", prec1, "\n") 

    #        sm = nn.Softmax()
            sm_pred = sm(y_pred.data.cpu().numpy()) # convert to numpy array
            y_norm_probs = sm_pred[:, 0]  # p(normal)
            y_pred_probs = sm_pred[:, 1]  # p(abnormal)
           # print('just before meta_data.append')

            meta_data.append(
                pd.DataFrame({
                    'img_filename': meta['img_filename'],
                    'y_true': label_v.data.view(1).cpu().numpy(),
                    'y_pred_probs': y_pred_probs,
                    'patient': meta['patient'].numpy(),
                    'study': meta['study'].numpy(),
                    'image_num': meta['image_num'].numpy(),
                    'encounter': meta['encounter'],
                }))


            # showing infomation of progress bar
            print("VALIDATION[{", i, "}/{", len(validate_loader), "}]", "acc=", prec1[0], ", average = ", validate_accuracy.average, "\n")

            image_var = Variable(images)
            label_var = Variable(target)
            last_iter_patient = cur_patient
            last_iter_study = cur_study

    y_prediction = model(image_var)
    label_v = label_var.float()
    y_pred = torch.sigmoid(y_prediction)
    #print("y_pred", y_pred, "\n")
    y_pred = y_pred.mean()
    y_pred = y_pred.expand(1,1)
    #print("y_pred", y_pred, "\n")
    label_v = Variable(torch.tensor(label_v[0]))
    label_v = label_v.expand(1,1)
    #print("label", label_v, "\n")
    
    loss = criterion(y_pred, label_v.view(-1, 1))
    label_v = label_v.long()
    validate_losses.update(loss.data[0], images.size(0))

    # update accuracy metric on the GPU
    _, prec1 = accuracy(y_pred.data, label_v.data, topk=(1, 1))
    #print("prec1 = ", prec1, "\n")
    validate_accuracy.update(prec1[0], images.size(0))
    print("VALIDATION[{", i, "}/{", len(validate_loader), "}]", "acc=", prec1[0], ", average = ", validate_accuracy.average, "\n")
    sm_pred = sm(y_pred.data.cpu().numpy()) # convert to numpy array
    y_norm_probs = sm_pred[:, 0]  # p(normal)
    y_pred_probs = sm_pred[:, 1]  # p(abnormal)
       # print('just before meta_data.append')
    meta_data.append(
         pd.DataFrame({
        'img_filename': meta['img_filename'],
        'y_true': label_v.data.view(1).cpu().numpy(),
        'y_pred_probs': y_pred_probs,
        'patient': meta['patient'].numpy(),
        'study': meta['study'].numpy(),
        'image_num': meta['image_num'].numpy(),
        'encounter': meta['encounter'],
    }))


        # showing infomation of progress bar
    pbar.set_description("VALIDATION[{}/{}]".format(i, len(validate_loader)))
    pbar.set_postfix(
        acc="{acc.value: .4f} ({acc.average: .4f})".format(acc = validate_accuracy),
        loss="{loss.value: .4f} ({loss.average: .4f})".format(loss = validate_losses))

    df = pd.concat(meta_data) # link tables

    # groupby: compute column sum mean
    ab = df.groupby(['encounter'])['y_pred_probs', 'y_true'].mean()

    ab['y_pred_round'] = ab.y_pred_probs.round()

    ab['y_pred_round'] = pd.to_numeric(ab.y_pred_round, downcast='integer')

    meter = AUCMeter()
    meter.add(ab.y_pred_probs, ab.y_true)
    roc_auc, _, _ = meter.value()
    print('roc_auc = ', roc_auc)
    #print('roc_auc = ', roc_auc_score(ab.y_true, ab.y_pred_round))
    #roc_auc = roc_auc_score(ab.y_true, ab.y_pred_round)
    return (f1_score(ab.y_true, ab.y_pred_round), roc_auc)

def save_checkpoint(state, is_best, filename='checkpoint.dense.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_BCE_batch8-1-2.dense.tar')

def sm(y_pred):
    batch_s, _ = y_pred.shape
    ret = numpy.zeros((batch_s, 2), dtype=np.float64)
    ret[:, 1]=y_pred[:, 0]
    ret[:, 0]=1-y_pred[:, 0]
    return ret

class DenseNet169(nn.Module):
    def __init__(self,outsize):
        super(DenseNet169,self).__init__()
        self.densenet169 = models.densenet169(pretrained = True)
        num_features = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(nn.Linear(num_features,outsize),nn.Sigmoid())
    def forward(self, x):
        x = self.densenet169(x)
        return x


"""Computes and stores the average and current value"""
class AverageMeter():
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, update_stride = 1):
        self.value = val
        self.sum += val * update_stride
        self.count += update_stride
        self.average = self.sum / self.count



def accuracy(y_prediction, y_groundtruth, topk=(1, )):
    batch_size = y_groundtruth.size(0)
    y_prediction = y_prediction-0.5
    y_prediction[y_prediction>=0]=1
    y_prediction[y_prediction<0]=0
    '''
    maxk = max(topk)
    batch_size = y_groundtruth.size(0)

    _, prediction = y_prediction.topk(maxk, 1, True, True)
    '''
    y_prediction = y_prediction.t()
    y_prediction = y_prediction.long()
    correct = y_prediction.eq(y_groundtruth.expand_as(y_prediction))
#    correct = prediction.eq(y_groundtruth.view(1, -1).expand_as(prediction))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


def runTrain(datadir,pathtomodel, pathsavemodel, workers,epochs,batch_size,lr,weight_decay):
    model = models.densenet201(pretrained = True)
   
    classes = 1 # here we can modify the classes of data
    model.classifier = nn.Linear(1920, classes) 
    model = torch.nn.DataParallel(model).cuda()
    #checkpoint = torch.load('model_best_BCE_batch8-1.dense.tar')
    checkpoint = torch.load(pathtomodel)
    model.load_state_dict(checkpoint['state_dict'])
    last_iter = checkpoint['best_roc_auc']
    print('last_iter = ', last_iter)
    
    # data loading
    data_dir =  datadir #'../mura/muraproc/' # data_dir_name is to be set
    train_dir = join(data_dir, 'train')
    train_csv = join(data_dir, 'train.csv')
    validate_dir = join(data_dir, 'valid')
    validate_csv = join(data_dir, 'valid.csv')
    test_dir = join(data_dir, 'test') # here remained for test
      
    # ensure that data loading is successful
    assert isdir(data_dir) and isdir(train_dir) and isdir(validate_dir) and isdir(test_dir)
    assert exists(train_csv) and isfile(train_csv) and exists(validate_csv) and isfile(validate_csv)

    # before feeding images into network, we normalize each image to same var and mean
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # scale image to 224*224, optional: random rotation
    train_transforms = transforms.Compose([
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
        # optional: random rotatation before training or not
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # call module MuraDataset, save it in train_data
    train_data = MuraDataset(train_csv, transform = train_transforms)

    # initial weights
    temp_weights = train_data.balanced_weights
    weights = torch.DoubleTensor(temp_weights)

    # here has several options: 
    # more optionals see at: https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/
    # sampler = torch.utils.data.sampler.SequentialSampler(train_data)
    # sampler = torch.utils.data.sampler.RandomSampler(train_data)  
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # define train_loader
    train_loader = data.DataLoader(
        train_data,
        batch_size = batch_size, # here we can adjust batch size
        # shuffle=True,  # here we can choose whether to shuffle
        num_workers = workers, # Load the data in parallel using multiprocessing workers
        sampler = sampler,
        pin_memory = True) # pin_memory is just for faster data transfers


    # define validate loader
    validate_loader = data.DataLoader(
        MuraDataset(validate_csv,
                    transforms.Compose([
#                        transforms.Resize(256),
#                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])),
        batch_size = 1, # here we can adjust batch size
        shuffle = False, # here we can choose whether to shuffle
        num_workers = workers, # Load the data in parallel using multiprocessing workers.
        pin_memory=True) # pin_memory is just for faster data transfers

    
    
    # define critical things for training 

    criterion = nn.BCELoss()
    # the second parameter in optim is learning rate
    # we can adjust the optim to SGD, etc.
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    best_validate_loss = 0
    best_roc_auc = 0
    # begin training
    
    for epoch in range(1, epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        validate_loss,roc_auc = validate(validate_loader, model, criterion, epoch)
        scheduler.step(validate_loss)
        # remember best accuracy and save checkpoint
        is_best = roc_auc > best_roc_auc
        best_roc_auc = max(roc_auc, best_roc_auc)
        best_validate_loss = max(validate_loss, best_validate_loss)
        save_checkpoint(
           { 'epoch': epoch + 1,
            'arch': 'densenet',
            'state_dict': model.state_dict(),
            'best_validate_loss': best_validate_loss,
            'best_roc_auc': best_roc_auc,
         }, is_best, pathsavemodel)

   
def runVal(datadir, pathtomodel):
    model = models.densenet201(pretrained = True)
   
    classes = 1 # here we can modify the classes of data
    model.classifier = nn.Linear(1920, classes) 
    model = torch.nn.DataParallel(model).cuda()
    #checkpoint = torch.load('model_best_BCE_batch8-1.dense.tar')
    checkpoint = torch.load(pathtomodel)
    model.load_state_dict(checkpoint['state_dict'])
    last_iter = checkpoint['best_roc_auc']
    print('last_iter = ', last_iter)
    
    # data loading
    data_dir =  datadir #'../mura/muraproc/' # data_dir_name is to be set
    train_dir = join(data_dir, 'train')
    train_csv = join(data_dir, 'train.csv')
    validate_dir = join(data_dir, 'valid')
    validate_csv = join(data_dir, 'valid.csv')
    test_dir = join(data_dir, 'test') # here remained for test
      
    # ensure that data loading is successful
    assert isdir(data_dir) and isdir(train_dir) and isdir(validate_dir) and isdir(test_dir)
    assert exists(train_csv) and isfile(train_csv) and exists(validate_csv) and isfile(validate_csv)

    # before feeding images into network, we normalize each image to same var and mean
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # scale image to 224*224, optional: random rotation
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # call module MuraDataset, save it in train_data
    train_data = MuraDataset(train_csv, transform = train_transforms)

    # initial weights
    temp_weights = train_data.balanced_weights
    weights = torch.DoubleTensor(temp_weights)

    # here has several options: 
    # more optionals see at: https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/
    # sampler = torch.utils.data.sampler.SequentialSampler(train_data)
    # sampler = torch.utils.data.sampler.RandomSampler(train_data)  
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # define validate loader
    validate_loader = data.DataLoader(
        MuraDataset(validate_csv,
                    transforms.Compose([
#                        transforms.Resize(256),
#                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])),
        batch_size = 1, # here we can adjust batch size
        shuffle = False, # here we can choose whether to shuffle
        num_workers = 2, # Load the data in parallel using multiprocessing workers.
        pin_memory=True) # pin_memory is just for faster data transfers

    
    
    # define critical things for training 

    criterion = nn.BCELoss()
    # we can adjust the optim to SGD, etc.

   # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    best_validate_loss = 0
    best_roc_auc = 0
    # begin training
    
    for epoch in range(1, 2):
        
        # evaluate on validation set
        validate_loss,roc_auc = validate(validate_loader, model, criterion, epoch)
    #    scheduler.step(validate_loss)
        # remember best accuracy and save checkpoint
        is_best = roc_auc > best_roc_auc
        best_roc_auc = max(roc_auc, best_roc_auc)
        best_validate_loss = max(validate_loss, best_validate_loss)
        '''save_checkpoint(
           { 'epoch': epoch + 1,
            'arch': 'densenet',
            'state_dict': model.state_dict(),
            'best_validate_loss': best_validate_loss,
            'best_roc_auc': best_roc_auc,
         }, is_best)
        '''

