# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:14:52 2017

@author: hnolte-101
"""


LIMIT_WIDTH_SMALL = 920
LIMIT_HEIGHT_SMALL = 700

LIMIT_WIDTH_NORMAL = 1150
LIMIT_HEIGHT_NORMAL = 930

LIMIT_WIDTH_LARGE = 1360
LIMIT_HEIGHT_LARGE = 1010
def check_resolution_for_icons(new_width,new_height,old_width= None,old_height = None, init_window = False):
    '''
    Checks for the resized resolution and returns:
        SMALL
        MEDIUM
        LARGE
    that can be used to reconfigure buttons with new ICONS...    
    '''
    if init_window:
        if new_width < LIMIT_WIDTH_SMALL:
            return 'NORM'
        elif new_width < LIMIT_WIDTH_NORMAL:
            return 'NORM'
        else:
            return 'LARGE'
    else:   
        if (new_width < LIMIT_WIDTH_SMALL and old_width > LIMIT_WIDTH_SMALL) or (new_height < LIMIT_HEIGHT_SMALL and old_height > LIMIT_HEIGHT_SMALL):
            return 'NORM'
        elif (new_width < LIMIT_WIDTH_NORMAL and old_width > LIMIT_WIDTH_NORMAL) or (new_width > LIMIT_WIDTH_SMALL and old_width < LIMIT_WIDTH_SMALL and new_width < LIMIT_WIDTH_LARGE) or (new_height < LIMIT_HEIGHT_NORMAL and old_height > LIMIT_HEIGHT_NORMAL) or  (new_height > LIMIT_HEIGHT_SMALL and old_height < LIMIT_HEIGHT_SMALL and new_height < LIMIT_HEIGHT_LARGE):
            
            return 'NORM'
        elif (new_width > LIMIT_WIDTH_NORMAL and old_width < LIMIT_WIDTH_NORMAL) or (new_height > LIMIT_HEIGHT_NORMAL and old_height < LIMIT_HEIGHT_NORMAL):
            return 'LARGE'
        else:
            return None 
    