from __future__ import absolute_import, division, print_function

import argparse
from os import getcwd
from os.path import exists, isdir, isfile, join

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
from Train import *
from dataloadertrain import *
from CAM import * 
from ml import *
from preprocess import *

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--data_dir', default='/data/wuyifan/muraproc', help='path to dataset')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--retrieval',default='image1.png', help = 'path to picture for image retrieval')
    parser.add_argument('--heatmapinput',default='image1.png',help ='path to heatmap input')
    parser.add_argument('--heatmapoutput', default='heatmap1.png',help= 'heatmap output path')
    parser.add_argument('--pathtotrainmodel',default='models/best_BCE.dense.tar',help = 'path to model for train and val')
    parser.add_argument('--pathtosavemodel',default='models/best.dense.tar',help ='path to save model')
    parser.add_argument('--retrievalpath', default='image1.png', help = 'path to img for retrieval')
    parser.add_argument('--modelretrieval', default='models/model_retrieval.tar', help ='path to model for retrieval')    
    args = parser.parse_args()
   # TrainModel(args.data_dir, args.pathtotrainmodel, args.workers, args.epochs, args.batch_size, args.lr, args.weight_decay)
   # runTrain(args.data_dir, args.pathtotrainmodel, args.workers, args.epochs, args.batch_size, args.lr, args.weight_decay)
    runVal(args.data_dir, args.pathtotrainmodel)
    GetHeatMap(args.heatmapinput, args.heatmapoutput, args.modelretrieval)
    answer_query(args.retrievalpath)
    
if __name__ == '__main__':
    main()
