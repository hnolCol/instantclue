# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:04:18 2017

@author: hnolte-101
"""
from sys import platform as _platform

def return_platform():
    if _platform == "linux" or _platform == "linux2":
       platform = 'LINUX'
    elif _platform == "darwin":
       platform = 'MAC'
    elif _platform == "win32":
       platform = 'WINDOWS'
    elif _platform == "win64":
        platform = 'WINDOWS'
        
    return platform    