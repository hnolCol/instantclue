"""
	""UTILITY FUNCTIONS""
    Instant Clue - Interactive Data Visualization and Analysis.
    Copyright (C) Hendrik Nolte

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
try:
	from matplotlib._color_data import TABLEAU_COLORS,XKCD_COLORS,CSS4_COLORS	
except:
	TABLEAU_COLORS,XKCD_COLORS,CSS4_COLORS = dict(), dict(), dict()
import seaborn as sns
from math import sqrt
import numpy as np
from collections import OrderedDict
from decimal import Decimal
GREY = "#efefef"
defaultFont = 'Verdana'
savedMaxColors = dict()


styleHoverScat = dict(visible=False, c = 'red', marker = 'o',
				markeredgecolors = 'black',
				markeredgewidth = 0.3)#

styleHoverScatter = dict(visible=False, c = 'red',
				s = 50,
				edgecolors = 'black',
				linewidths = 0.3)#


def match_color_to_uniqe_value(uniqueValues, colorMap):
	'''
	Unique values
	Needed for correct color when missing values occur / categorical plotting
	'''
	if isinstance(uniqueValues,np.ndarray):
		uniqueValues = uniqueValues.tolist()

	if isinstance(uniqueValues[0],list):
		uniqueValues = [tuple(x) for x in uniqueValues]
	numbUniqueValues = len(uniqueValues)

	colorMap = sns.color_palette(colorMap,numbUniqueValues)
	colorDict = OrderedDict(zip(uniqueValues,colorMap))
	return colorDict

def get_color_category_dict(dfClass,categoricalColumn,colorMap,userDefinedColors,naColor = GREY,naString = '-'):
	'''
	This is needed for updating the color in scatter plots with same colors
	Retrns a dict of : Categorical level - color
	'''
	if isinstance(categoricalColumn,str):
		categoricalColumn = [categoricalColumn]
	numCategoricalColumn = len(categoricalColumn)
	if  numCategoricalColumn == 1:

		uniqueCategories = dfClass.get_unique_values(categoricalColumn[0])
		numbUniqueCategories = uniqueCategories.size
		rgbColors = sns.color_palette(colorMap, numbUniqueCategories)
		colorMapDict = OrderedDict(zip(uniqueCategories ,rgbColors))
		colorMapDict = replace_nan_in_color_dict(colorMapDict,naColor,
							additionalNanLevels = [naString])

	else:
		collectUniqueCategories = []
		for categColumn in categoricalColumn:

			catUnique = dfClass.get_unique_values(categColumn)
			collectUniqueCategories.append(catUnique.tolist())

		theoreticalCombinations = list(itertools.product(*collectUniqueCategories))
		combinationsInData = dfClass.df[categoricalColumn].apply(tuple,axis=1)
		uniqueCategories = combinationsInData.unique()
		numbUniqueCombData = uniqueCategories.size

		rgbColors = sns.color_palette(colorMap,numbUniqueCombData)

		colorMapDict = OrderedDict(zip(uniqueCategories ,rgbColors))

		colorMapDict = replace_nan_in_color_dict(colorMapDict,naColor,
							additionalNanLevels = [(naString,)*numCategoricalColumn])

	colorMapDictRaw = colorMapDict.copy()
	for category in uniqueCategories:
			if category in userDefinedColors:
				colorMapDict[category] = userDefinedColors[category]

	layerColorDict = build_rgb_layer_dict(rgbColors,GREY,naColor)

	return colorMapDict, layerColorDict, colorMapDictRaw


def split_except_brac(self,s):
    parts = []
    bracket_level = 0
    current = []
    # trick to remove special-case of trailing chars
    for c in (s + " "):
        if c == " " and bracket_level == 0:
            parts.append("".join(current))
            current = []
        else:
            if c == "{":
                bracket_level += 1
            elif c == "}":
                bracket_level -= 1
            current.append(c)
    return parts

def xLim_and_yLim_delta(ax):

    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    delta_x = xmax-xmin
    delta_y = ymax-ymin
    return delta_x,delta_y




def calculate_new_ylim_from_data(data, force_ylim_NULL = False):
    ""
    get_max_val = data.max().max()
    get_min_val = data.min().min()
    add = 0.12
   
    y_max = round(get_max_val+get_max_val*add,2)
    if force_ylim_NULL:
        y_min = 0
    else:
        y_min = round(get_min_val-abs(get_min_val*add),2)

    return (y_min,y_max)


def get_elements_from_list_as_string(itemList, addString = '',newLine = False, maxStringLength = 40):
	'''
	'''
	if isinstance(itemList,str):
		return itemList
	for n,item in enumerate(itemList):
		if maxStringLength is not None:
			item = str(item)[:maxStringLength]
		if n == 0:
			if newLine:
				addString = '{}\n{}'.format(addString,item)
			else:

				addString = '{}{}'.format(addString,item)
		else:
			addString = '{}, {}'.format(addString,item)

	return addString




def return_readable_numbers(number):
     """Returns number as string in a meaningful and readable way to omit to many digits"""
     
     orgNumber = number
     number = np.abs(number)
     if number == 0:
     	new_number = 0.0
     elif number < 0.001:
     		new_number = '{:.2E}'.format(number)
     elif number < 0.1:
     		new_number = round(number,4)
     elif number < 1:
     		new_number = round(number,3)
     elif number < 10:
     		new_number = round(number,2)
     elif number < 200:
     		new_number = float(Decimal(str(number)).quantize(Decimal('.01')))
     elif number < 10000:
     		new_number = round(number,0)
     else:
     		new_number = '{:.2E}'.format(number)
     if orgNumber >= 0:
     	return new_number
     else:
     	return new_number * (-1)


def get_max_colors_from_pallete(cmap):
    '''
    '''

    if cmap in ["Greys","Blues","Greens","Purples",'Reds',"BuGn","PuBu","PuBuGn","BuPu",
    	"OrRd","BrBG","PuOr","Spectral","RdBu","RdYlBu","RdYlGn","viridis","inferno","cubehelix"]:
        n = 60
    elif cmap in ['Accent','Dark2','Pastel2','Set2','Set4']:
        n=8
    elif cmap in ['Tenenbaums','Darjeeling Limited','Moonrise Kingdom','Life Acquatic']:
    	n = 5
    elif cmap in ['Paired','Set3']:
        n = 12
    elif cmap in ['Pastel1','Set1']:
        n = 9
    elif cmap == 'Set5':
        n = 13
    elif cmap in savedMaxColors:
    	n = savedMaxColors[cmap]
    else:
    	n = 0
    	col = []
    	for color in sns.color_palette(cmap,n_colors=60):
    		if color not in col:
    			col.append(color)
    			n += 1
    		else:
    			break
    	savedMaxColors[cmap] = n


    cmap = ListedColormap(sns.color_palette(cmap, n_colors = n ,desat = 0.85))

    return cmap
