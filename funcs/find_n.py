# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:49:18 2017

@author: hnolte-101
"""
import numpy as np
from math import sqrt
from operator import itemgetter
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
def find_nearest_index(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
def distance(co1, co2):
        return sqrt(pow(abs(co1[0] - co2[0]), 2) + pow(abs(co1[1] - co2[1]), 2))
    
def closest_coord_idx(list_, coord):
            if coord is not None:

            	dist_list = [distance(co,coord) for co in list_]
            	idx = min(enumerate(dist_list),key=itemgetter(1))
            	return idx
