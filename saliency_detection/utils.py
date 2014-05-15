#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    image processing utilities.
"""

import sys
import os
import math
import itertools
import cv2 as cv
import numpy as np


class OpencvIo:
    def __init__(self):
        self.__util = Util()

    # 读取文件并返回，用cv.imread读
    def imread(self, path, option=1):
        try:
            # 判断文件是否存在
            if not os.path.isfile(os.path.join(os.getcwd(), path)):
                raise IOError('File is not exist')
            src = cv.imread(path, option)
        except IOError:
            raise
        except:
            print 'Arugment Error : Something wrong'
            sys.exit()
        return src

    # 显示图像，调用opencv的imshow, 最后加了销毁所有Window
    def imshow(self, src, name='a image'):
        cv.imshow(name, src)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # 把一个数组（可in的数据结构）里的图像全显示，窗口名是从0开始编号。
    def imshow_array(self, images):
        name = 0
        for x in images:
            cv.imshow(str(name), np.uint8(self.__util.normalize_range(x)))
            name = name + 1
        cv.waitKey(0)
        cv.destroyAllWindows()


class Util:
    # 把src里的像素值归一化到 (begin,end)
    def normalize_range(self, src, begin=0, end=255):
        # 把输出的二维数组 normalize 到 begin - end
        dst = np.zeros((len(src), len(src[0])))
        amin, amax = np.amin(src), np.amax(src)
                    # 这个 itertools.product 比较虎
        for y, x in itertools.product(xrange(len(src)), xrange(len(src[0]))):
            if amin != amax:
                # 归一化公式，（V - min ) * range_new / range_old + begin
                dst[y][x] = (src[y][x] - amin) * (end - begin) / (amax - amin) + begin
            else:
                # 如果值都一样，取新范围的中值
                dst[y][x] = (end + begin) / 2
        return dst

    # 把src里的像素值归一化到 （0,1）然后再根据最值和平均极值的差对原像素值进行一个数乘
    # Itti PAMI 1998 讲了道理，因为算法总共算了42个Saliency Map，要把它们融合成一个，为
    # 防止比较NB的Sliency被平均掉，做的这种后处理。可以突出有单一高Saliency的情况，抵制
    # 有多个差不多的Saliency的情况。
    def normalize(self, src):
        # 1, 调用上面的 normalize_range 把 src 归一化到 0-1
        # 2, 找图像中的极大值点，如果有极大值点，就进行一个后处理
        #   后处理是乘一个数，这个数是 （最大值 - 极大值平均）^2
        src = self.normalize_range(src, 0., 1.)
        amax = np.amax(src)
        maxs = []

        for y in xrange(1, len(src) - 1):
            for x in xrange(1, len(src[0]) - 1):
                val = src[y][x]
                if val == amax:
                    continue
                if val > src[y - 1][x] and val > src[y + 1][x] and val > src[y][x - 1] and val > src[y][x + 1]:
                    maxs.append(val)

        if len(maxs) != 0:
            # src 是opencv的 imread 读出来的结果。应该是定义了 *= 运算符
            src *= math.pow(amax - (np.sum(maxs) / np.float64(len(maxs))), 2.)

        return src
