import time
import os
from PIL import Image 
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from readdir import get_model
from feature import densemodel
from preprocess import preprocess
import torchvision.transforms as transforms
queryfile = 'out.txt'
text_dir = 'query_weight.txt'
weightlen = 81536

def similarity(vec):
    data = list()
    cnt = 0
    path = []
    min_dis = 1e10
    with open(queryfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split()
            thispath = line[0]
            line.remove(line[0])
            line = np.array(line)
            line = line.astype(float)
            tmp_dis=0
            for i in range(weightlen):
                tmp = np.square(vec[i]-line[i])
                tmp_dis += tmp 
            #print(tmp_dis)
            if tmp_dis < min_dis:
                min_dis = tmp_dis
                minpath = thispath
    #print(min_dis)
    #print(minpath)
    f.close()
    return minpath 


def feature_extraction(filename_in):
    model, transform = get_model()
    filename_out = "query_image.png"
    preprocess(filename_in, filename_out)
    img = Image.open(filename_out).convert('RGB')
    img = transform(img)
    img_PIL = transforms.ToPILImage()(img).convert('RGB')
    model_feature = model.module.features                
    net = densemodel(model_feature, img_PIL)
    weight = net.extract_lastlayer()
    weight1 = weight[0].data.cpu().numpy()
    f = open(text_dir,"w")
    for i in range(1920):
        b = 0.0
        for j in range(7):
            a = sum(weight1[i][j])
            b += a
        b /= 49 
        f.write(str(b)+' ')
    f.close()
    return weight1


def output_img(path):
    img=cv2.imread(path, cv2.IMREAD_COLOR)
    plt.imshow(img)
    plt.show()


def answer_query(query_dir):
    #print('query_dir:', query_dir)
    vector = feature_extraction(query_dir)
    os.system('g++ -o exe weight.cpp')
    time.sleep(1)
    os.system('./exe')

    #img_name = answer_query(sys.argv[1])
    #print(img_name)
    
