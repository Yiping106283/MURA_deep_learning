# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 01:53:21 2018

@author: huomingjia
"""


from __future__ import absolute_import, division, print_function

import re 
import numpy as np
import pandas as pd
import torch.utils.data as data
from os import getcwd
from os.path import join
from PIL import Image

class MuraDataset(data.Dataset):
#define regular language to recognize filename
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')
 
    def __init__(self, csv_f, transform=None, download=False):
        self.df = pd.read_csv(csv_f, names=['img', 'label'], header=None)
        self.imgs = self.df.img.values.tolist()
        self.labels = self.df.label.values.tolist()
        self.transform = transform 
        # following datasets/folder.py's weird convention here...
        self.samples = [tuple(x) for x in self.df.values]
        self.classes = np.unique(self.labels)  # number of unique classes
        self.balanced_weights = self.balance_class_weights()

    def __len__(self):
        return len(self.imgs)

    # use RE to get the index of imgs
    def _parse_patient(self, filename):
        return int(self._patient_re.search(filename).group(1))

    def _parse_study(self, filename):
        return int(self._study_re.search(filename).group(1))

    def _parse_image(self, filename):
        return int(self._image_re.search(filename).group(1))

    def _parse_study_type(self, filename):
        return self._study_type_re.search(filename).group(1)

    def __getitem__(self, index):
        img_name = join(self.imgs[index])
        img_name = img_name[img_name.find("/"):]
        img_name = "/data/wuyifan/muraproc"+img_name
       # print(self.imgs[index])
        patient = self._parse_patient(img_name)
        study = self._parse_study(img_name)
        image_num = self._parse_image(img_name)
        study_type = self._parse_study_type(img_name)

        # todo(bdd) : inconsistent right now, need param for grayscale / RGB
        # todo(bdd) : 'L' -> gray, 'RGB' -> Colors
        image = Image.open(img_name).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        meta_data = {
            'y_true': label,
            'img_filename': img_name,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient, study)
        }
        return image, label, meta_data

    def balance_class_weights(self):
        cnt_classes = len(self.classes)
        weight_per_class = [0.] * cnt_classes

        cnt_imgs = [0] * cnt_classes
        for item in self.samples:
            cnt_imgs[item[1]] += 1
        N = float(sum(cnt_imgs))
        for i in range(cnt_classes):
            weight_per_class[i] = N / float(cnt_imgs[i])
        # Here we balance the total positive and negative weights to be equal. 
        # Sum weights of each class = N

        weight = [0] * len(self.samples)
        for index, val in enumerate(self.samples):
            weight[index] = weight_per_class[val[1]]
        return weight
