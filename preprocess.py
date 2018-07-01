import argparse
import cv2
from os import mkdir, makedirs
from os.path import exists
import numpy as np
import csv
import sys

from edgefinder import *


def reverse_color(img):
    return 255-img

def preprocess(filename_in, filename_out):

    img = cv2.imread(filename_in, cv2.IMREAD_GRAYSCALE)
    m, n = img.shape

    img = crop_center_with_pixel(cv2.resize(img, (320, 320)), 280, 280)
    edge_finder = EdgeFinder(img, filter_size=13, threshold1=0, threshold2=8)

    '''
    (head, tail) = os.path.split(filename)

    (root, ext) = os.path.splitext(tail)
    
    smoothed_filename = root +"-smoothed" + ext
    edge_filename = root + "-edges" + ext

    cv2.imwrite(smoothed_filename, edge_finder.smoothedImage())
    cv2.imwrite(edge_filename, edge_finder.edgeImage())
    '''

    x, y = find_act(edge_finder.edgeImage())

    create_path(filename_out)
    cv2.imwrite(filename_out, img[x: x+crop_size, y:y+crop_size])


def csv_process(in_dir, out_dir, csv_name):
    with open(in_dir+csv_name, 'r') as csvfile:
        img_reader = csv.reader(csvfile)
        for row in img_reader:
            tmp = row[0].find("/")
            if in_dir[-1]=='/':
                tmp += 1
            filename = row[0][tmp:]
            for i in range(1, 10):
                filename_in = in_dir + filename + "image" + str(i) + ".png"
                filename_out = out_dir + filename + "image" + str(i) + ".png"
                #print(filename_out)
                if not exists(filename_in):
                    break
                preprocess(filename_in, filename_out)

def create_path(s):
    p = s.rfind("/")
    mk = s[: p]
    if not exists(mk):
        makedirs(mk)


def main():
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    csv_name = sys.argv[3]
    csv_process(in_dir, out_dir, csv_name)

if __name__ == '__main__':
    main()