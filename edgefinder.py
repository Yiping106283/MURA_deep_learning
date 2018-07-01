import sys
import cv2
import numpy as np
import os

crop_size = 224

def crop_center_with_pixel(img,cropx,cropy):
    y,x= img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx]


def crop_center_with_scale(img,scalex,scaley):
    y,x= img.shape
    cropx = int(x*scalex)
    cropy = int(y*scaley)
    return crop_center_with_pixel(img, cropx, cropy)

def find_act(img):
    img = img.astype(np.int32)
    kernel = np.ones((crop_size, crop_size), img.dtype)
    cnt = np.zeros((crop_size, crop_size), dtype = img.dtype)
    cnt = cv2.filter2D(img, -1, kernel, anchor=(0, 0))
    s = np.argmax(cnt)
    n, m = img.shape
    x = int(s/n)
    y = int(s%m)
    if x+crop_size>n:
        x = n-crop_size
    if y+crop_size>m:
        y = m-crop_size

    return x, y

def reverse_detector(img):
    circle_finder = EdgeFinder(img, filter_size=13, threshold1=50, threshold2=115)


class EdgeFinder:
    def __init__(self, image, filter_size=1, threshold1=0, threshold2=0, require_blur = False):
        self.image = image
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self.require_blur = require_blur
        self._render()
    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size

    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def _render(self):
        if self.require_blur:
            self._smoothed_img = cv2.GaussianBlur(self.image, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        else:
            self._smoothed_img = self.image
        self._edge_img = cv2.Canny(self._smoothed_img, self._threshold1, self._threshold2)

def main():
    filename_in = sys.argv[1]

    (head, tail) = os.path.split(filename_in)
    (root, ext) = os.path.splitext(tail)

    filename_out = root + "-circle" + ext

    img = cv2.imread(filename_in, cv2.IMREAD_GRAYSCALE)
    
    edge_finder = EdgeFinder(img, filter_size=13, threshold1=50, threshold2=115)

    cv2.imwrite(filename_out, edge_finder.edgeImage())

if __name__ == '__main__':
    main()

