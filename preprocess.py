import argparse
import cv2
import os
import numpy as np
import csv

from edgefinder import EdgeFinder

crop_size = 224

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

    cv2.imwrite(filename_out, img[x: x+crop_size, y:y+crop_size])


def csv_process(in_dir, out_dir):
    with open(in_dir+'test.csv', 'r') as csvfile:
        img_reader = csv.reader(csvfile)
        for row in img_reader:
            #tmp = row[0].find("/")+1
            #filename = row[0, tmp:], label = row[1]
            filename_in = in_dir + row[0]
            filename_out = out_dir + row[0]
            preprocess(filename_in, filename_out)



def main():
    csv_process("./", "./")

if __name__ == '__main__':
    main()