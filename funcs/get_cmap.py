# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:42:57 2017

@author: hnolte-101
"""
from matplotlib.colors import ListedColormap
import seaborn as sns
Set4 = ['#D1D3D4','#6D6E71','#EE3124','#FCB74A','#2E5FAC','#9BD5F4',
          '#068135','#91CA65']
Set5 = ['#7F3F98','#2E5FAC','#27AAE1','#9BD5F4','#017789','#00A14B',
          '#91CA65','#ACD9B2','#FFDE17','#FCB74A','#F26942',
           '#EE3124','#BE1E2D']

sns.palettes.SEABORN_PALETTES['Set4'] = Set4
sns.palettes.SEABORN_PALETTES['Set5'] = Set5

def get_max_colors_from_pallete(cmapName):
    if cmapName in ["Greys","Blues","Greens","Purples",'Reds',"BuGn","PuBu","PuBuGn","BuPu","OrRd","BrBG","PuOr","Spectral","RdBu","RdYlBu","RdYlGn"]:
        n = 60
    elif cmapName in ['Accent','Dark2','Pastel2','Set2','Set4']:
        n=8
    elif cmapName in ['Paired','Set3']:
        n = 12
    elif cmapName in ['Pastel1','Set1']:
        n = 9
    elif cmapName == 'Set5':
        n = 13
    
    cmap = ListedColormap(sns.color_palette(cmapName, n_colors = n ,desat = 0.85))  
    
    return cmap