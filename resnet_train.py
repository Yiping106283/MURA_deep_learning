# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:40:27 2018

@author: mayiping
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from torch.optim.lr_scheduler import ReduceLROnPlateau

from os.path import exists, isdir, isfile, join
from os import getcwd
import shutil
from tqdm import tqdm

'''
from dataloader import MuraDataset # here load module dataloader
'''

# here we define all the global var
best_validate_loss = 0

def work():
    model = models.resnet18(pretrained = True)
    
    #extract fc layer's parameters 
    fc_features = model.fc.in_features  
    
    classes = 2 # here we can modify the classes of data
    model.fc = nn.Linear(fc_features, classes) 
    
    # data loading
    data_dir = join(getcwd(), data_dir_name) # data_dir_name is to be set
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
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # optional: random rotatation before training or not
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
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
        batch_size = 50, # here we can adjust batch size
        # shuffle=True,  # here we can choose whether to shuffle
        num_workers = 4, # Load the data in parallel using multiprocessing workers
        sampler = sampler,
        pin_memory = True) # pin_memory is just for faster data transfers


    # define validate loader
    validate_loader = data.DataLoader(
        MuraDataset(validate_csv,
                    transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])),
        batch_size = 50, # here we can adjust batch size
        shuffle = False, # here we can choose whether to shuffle
        num_workers = 4, # Load the data in parallel using multiprocessing workers.
        pin_memory=True) # pin_memory is just for faster data transfers

    
    
    # define critical things for training 

    criterion = nn.CrossEntropyLoss()
    # the second parameter in optim is learning rate
    # we can adjust the optim to SGD, etc.
    optimizer = optim.Adam(model.parameters(), 1e-4, weight_decay = 1e-4)
    
    
    '''
    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    This scheduler reads a metrics quantity and
    if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.
    See more at https://pytorch.org/docs/stable/optim.html
    '''
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, verbose=True)


    # begin training
    start_epoch = 1
    epochs = 20

    for epoch in range(start_epoch, epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        validate_loss = validate(validate_loader, model, criterion, epoch)
        scheduler.step(validate_loss)
        # remember best accuracy and save checkpoint
        is_best = validate_loss > best_validate_loss  # best_validate_loss should be a global var
        best_validate_loss = max(validate_loss, best_validate_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet',
            'state_dict': model.state_dict(),
            'best_validate_loss': best_validate_loss,
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    # this can be modified, sometimes we set it to zero
    train_accuracy = DataContainer()
    train_losses = DataContainer()

    # pbar is progress bar, tqdm is a module for showing progress bar
    pbar = tqdm(train_loader)

    # train process begin
    for i, (images, target, meta) in enumerate(pbar):
        #target = target.cuda(async=True) # moving data to gpu

        # turing image and target to torchvariable
        image_var = Variable(images)
        label_var = Variable(target)

        # pass this batch through our model and get y_pred
        y_prediction = model(image_var)

        # update loss metric
        loss = criterion(y_prediction, label_var)
        train_losses.update(loss.data[0], images.size(0))

        # update accuracy metric
        _, prec1 = accuracy(y_prediction.data, target, topk=(1, 1))
        train_accuracy.update(prec1[0], images.size(0))
        
        '''
        Here we can change the accuracy function
        Official implementation: see at
        https://github.com/Atcold/pytorch-CortexNet/blob/master/image-pretraining/main.py
        # measure accuracy and record loss
        top1 = DataContainer()
        top5 = DataContainer()
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        '''

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

    validate_accuracy = DataContainer()
    validate_losses = DataContainer()
    meta_data = []

    # show the progress bar
    pbar = tqdm(validate_loader)

    for i, (images, target, meta) in enumerate(pbar):
        #target = target.cuda(async=True) move it to gpu
        image_var = Variable(images, volatile=True)
        label_var = Variable(target, volatile=True)

        y_pred = model(image_var)
        # udpate loss metric
        loss = criterion(y_pred, label_var)
        validate_losses.update(loss.data[0], images.size(0))

        # update accuracy metric on the GPU
        _, prec1 = accuracy(y_pred.data, target, topk=(1, 1)) 
        validate_accuracy.update(prec1[0], images.size(0))        
         
        '''
        Here we can change the accuracy function
        Official implementation: see at
        https://github.com/Atcold/pytorch-CortexNet/blob/master/image-pretraining/main.py
        # measure accuracy and record loss
        top1 = DataContainer()
        top5 = DataContainer()
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        '''
        
        sm = nn.Softmax()
        sm_pred = sm(y_pred).data.numpy() # convert to numpy array

        y_norm_probs = sm_pred[:, 0]  # p(normal)
        y_pred_probs = sm_pred[:, 1]  # p(abnormal)

        meta_data.append(
            pd.DataFrame({
                'img_filename': meta['img_filename'],
                'y_true': meta['y_true'].numpy(),
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
    # to_numeric: change the column type
    ab['y_pred_round'] = pd.to_numeric(ab.y_pred_round, downcast='integer') 

    '''
    https://blog.csdn.net/sinat_26917383/article/details/75199996
    accuracy_score(y_true, y_pred, normalize=False)  # 类似海明距离，每个类别求准确后，再求微平均
    metrics.f1_score(y_true, y_pred, average='weighted')  
    
    from sklearn.metrics import cohen_kappa_score
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    cohen_kappa_score(y_true, y_pred)
    '''
    print(cohen_kappa_score(ab.y_true, ab.y_pred_round))
    return f1_score(ab.y_true, ab.y_pred_round)

def save_checkpoint(state, is_best, filename = 'checkpoint.res.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.res.tar')

"""Computes and stores the average and current value"""
def DataContainer():
    
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


"""Computes the precision@k for the specified values of k"""
''' official pytorch implementation, see at
https://github.com/Atcold/pytorch-CortexNet/blob/master/image-pretraining/main.py
'''
def accuracy(y_prediction, y_groundtruth, topk=(1, )):
    maxk = max(topk)
    batch_size = y_groundtruth.size(0)

    _, prediction = y_prediction.topk(maxk, 1, True, True)
    prediction = prediction.t()
    correct = prediction.eq(y_groundtruth.view(1, -1).expand_as(prediction))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

    