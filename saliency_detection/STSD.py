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
from utils import OpencvIo


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
            aveLocal = np.mean(src[y-2:y+3, x-2:x+3])
            SLmap[y][x] = np.power(np.abs(aveLocal - aveAll),2)
        return SLmap
class DirectionalCoherence:
    def __init__(self,src):
        self.SDmaps = self.__makeSD(src)
    def __makeSD(self,src):
        its = src['intensity']
        return map(self.__SDOneFrame, its)
        #gds = np.gradient(its)
        #io.imshow_array([gds[0]])
    def __SDOneFrame(self, src):
        util = Util()
        io = OpencvIo()
        [grdy,grdx] = np.gradient(src)

        CoherenceMap = np.zeros(src.shape)
        SDmap = np.zeros(src.shape)
        windowsize = 7
        rgy = xrange(windowsize/2, len(src) - windowsize/2)
        rgx = xrange(windowsize/2, len(src[0]) - windowsize/2)

        for (y,x) in itertools.product(rgy,rgx):
            wgrdx = grdx[y - windowsize/2: y + (1+windowsize)/2 , x-windowsize/2: x+ (1+windowsize)/2]
            wgrdy = grdy[y - windowsize/2: y + (1+windowsize)/2 , x-windowsize/2: x+ (1+windowsize)/2]
            T_s = np.array( [ [np.sum(np.power(wgrdx,2)),       np.sum(wgrdy*wgrdx)],
                              [np.sum(wgrdy*wgrdx),       np.sum(np.power(wgrdy,2))]
                            ]
                           )
            eigvs = np.linalg.eigvals(T_s)
            CoherenceMap[y][x] = np.power(eigvs[0] - eigvs[1], 2)

        for (y,x) in itertools.product(rgy,rgx):
            centerval = CoherenceMap[y][x]
            windowsize = 7
            wCoh = CoherenceMap[y - windowsize/2: y + (1+windowsize)/2, x-windowsize/2: x+ (1+windowsize)/2]
            SDmap[y][x] = np.sum(np.abs(wCoh - centerval))

        return SDmap
        #io.imshow_array(gds)

class SaliencyMap:
    def __init__(self, src):
        util = Util()
        self.pimgs = PrepareImgs(src)
        self.SLs   = ContrastL(self.pimgs.maps)
        self.SDs    = DirectionalCoherence(self.pimgs.maps)
