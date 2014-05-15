#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Saliency Map.
"""
import math
import itertools
import cv2 as cv
import numpy as np
from utils import Util


class PrepareImgs:
    def __init__(self, src):
        self.maps = self.__luminal_contrast(src)

    def __luminal_contrast(self, src):
        maps = {'intensity': []}

        b, g, r = cv.split(src)

        buf_its = np.zeros( b.shape)

        for y, x in itertools.product(xrange(len(b)), xrange(len(b[0]))):
            buf_its[y][x] = self.__get_intensity(b[y][x], g[y][x], r[y][x])
        
        maps['intensity'].append(buf_its)
        return maps

    def __get_intensity(self, b, g, r):
        return (np.float64(b) + np.float64(g) + np.float64(r)) / 3.

class ContrastL:
    def __init__(self, src):
        self.SLmaps = self.__makeSL(src)

    def __makeSL(self,src):
        its = src['intensity']
        return map(self.__SLOneFrame, its)
    def __SLOneFrame(self,src):
        aveAll = np.mean(src)

        SLmap = np.zeros(src.shape)

        for (y,x) in itertools.product(xrange(2, len(src) -2), xrange(2, len(src[0]) -2)):
            abc = src[y-2:y+3, x-2:x+3]
            aveLocal = np.mean(src[y-2:y+3, x-2:x+3])
            SLmap[y][x] = np.power(np.abs(aveLocal - aveAll),4)
        return SLmap


class SaliencyMap:
    def __init__(self, src):
        util = Util()
        self.pimgs = PrepareImgs(src)
        self.SLs   = ContrastL(self.pimgs.maps)
