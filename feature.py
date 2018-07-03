import torch
#import matplotlib.image as mpimg
import numpy as np
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
#import cv2
import sys
from PIL import Image

class densemodel():
    def __init__(self, model, img):
        self.model = model
        self.model.eval()
        self.image = self.image_for_pytorch(img)

    def show(self):
        x = self.image
        x = x.cuda(async=True)
        cnt = 0
        for index, layer in enumerate(self.model):
            print(index,layer)
            #print(x)  # print every layer value
            print(x.shape)
            x = layer(x)
            cnt += 1
            print(cnt)
			
    def extract_lastlayer(self):
        x = self.image
        x=x.cuda(async=True)
        cnt = 0
        for index, layer in enumerate(self.model):
            #print(index,layer)
            if cnt == 11:
                #return x
                #print(x.shape)
                #print(index,layer)
                return x
            x = layer(x)
            cnt = cnt + 1
    
    def image_for_pytorch(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]  
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                 std=(0.229, 0.224, 0.225))
        ])
        imgres = transform(img)
        imgres = Variable(torch.unsqueeze(imgres, dim=0), requires_grad=True)
        return imgres



if __name__=="__main__":
    #print("Hi~(^_^), Please type in the location of the file to be retrieved here, such as \'..\data\image1.png\':")
    #image = Image.open('image1.png').convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.Resize(227),
        #transforms.CenterCrop(224),
        # optional: random rotatation before training or not
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30), 
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    img = Image.open(sys.argv[1]).convert('RGB')
    #img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = train_transforms(img)
    img1 = transforms.ToPILImage()(img).convert('RGB')
    model = models.densenet169(pretrained = True).features
    #model.classifier = nn.Linear(1664, 2)
    net = densemodel(model, img1)
    net.show()
    #net.extract_lastlayer()
