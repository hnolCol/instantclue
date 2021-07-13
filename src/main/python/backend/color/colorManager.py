

import seaborn as sns
from collections import OrderedDict
from .colorHelper import ColorHelper
from ..utils.misc import replaceKeyInDict
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm
import numpy as np
import pandas as pd



def rgbToHex(color):
	'''
	from rgb to hex
	'''
	if color is None or color == 'k':
		#ugly - think better
		return '#000000'
	if '#' in color:
		return color

	if isinstance(color,tuple):
		y = tuple([int(float(z) * 255) for z in color])
		hexCol = "#{:02x}{:02x}{:02x}".format(y[0],y[1],y[2])
	elif isinstance(color,list):
		y = tuple([int(float(z) * 255) for z in color[:3]])
		hexCol = "#{:02x}{:02x}{:02x}".format(y[0],y[1],y[2])
#	elif 'xkcd:'+str(color) in XKCD_COLORS:
#		hexCol = XKCD_COLORS['xkcd:'+str(color)]
#	elif 'tab:'+str(color) in TABLEAU_COLORS:
#		hexCol = TABLEAU_COLORS['tab:'+str(color)]
#	elif color in namedColors:
#		hexCol = namedColors[color]
#	else:
#		try:
#			hexCol = matplotlib.colors.to_hex(color)
#		except:
#			hexCol = 'error'


	return hexCol

class ColorManager(object):
    ""

    def __init__(self, sourceData, colorMap = "RdYlBu", nanColor = "#efefef", scatterSize = 50):
        ""
        self.sourceData = sourceData
        self.colorMap = colorMap
        self.twoColorMap = "RdYlBu"
        self.hclustSizeColorMap = "Greys"
        self.colorHelper = ColorHelper()
        self.hclustColorMap = "RdYlBu"
        self.hclustClusterColorMap = "Paired"
        self.quickSelectColorMap = "Paired"
        self.countplotLabelColorMap = "Blues"
        self.nanColor = nanColor
        self.desat = 0.75
        self.alpha = 0.85
        self.defaultScatterSize = scatterSize
        self.savedMaxColors = {}

  

    def buildColorLayerMap(self,rgbColors):
        ""
        layerColorDict = {}
        for n,color in enumerate(rgbColors):
            layerColorDict[color] = n+1

        layerColorDict[self.nanColor] = -2

        return layerColorDict


    def mapCategoricalDataToColors(self,  colorMap = None):
        ""

        #define colorMap
        colorMap = self.colorMap if colorMap is None else colorMap

    def getColorByDataIndex(self,dataID,categoricalColumn):
        ""
        if dataID in self.sourceData.dfs:
            
            try:
                uniqueValues = self.sourceData.getUniqueValues(dataID, categoricalColumn)
                #nColors = uniqueValues.size
                #colorPalette = sns.color_palette(self.colorMap,nColors,desat=self.desat).as_hex()
                #print(checkedLabels)
                colorMapDict, layerColorDict = self.createColorMapDict(uniqueValues, as_hex=True)
                colorData = self.sourceData.dfs[dataID][categoricalColumn].map(colorMapDict)
                colorData["layer"] = colorData.map(layerColorDict)
                return colorData
            except Exception as e:
                print(e)


    def colorDataByMatch(self,dataID,columnName, checkedLabels = None,checkedDataIndex = None, splitString = None, checkedSizes = None, userColors = None, colorMapName = None):
        ""
        if dataID in self.sourceData.dfs:
            if colorMapName is None or not hasattr(self,colorMapName):
                colorMapName = "colorMap"
            colorMap = getattr(self,colorMapName)
            nRow = self.sourceData.dfs[dataID].index.size

            colorData = pd.DataFrame([self.nanColor]*nRow, 
                                    index = self.sourceData.dfs[dataID].index, 
                                    columns=["color"])

            colorData["layer"] = [-1]*nRow
            colorData["size"] = [self.defaultScatterSize] * nRow
            if checkedDataIndex is not None:    
                
                #reset colorlabeling e.g no selection
                boolChecked = checkedDataIndex == True
                checkedData = checkedDataIndex[boolChecked]
                nColors = checkedData.size
                colorPalette = sns.color_palette(colorMap,nColors,desat=self.desat).as_hex()

                if nColors == 0:
                    return colorData, None, None
                else:
                    #set all data first to nanColor
                    #this needs to be faster
                    colorData.loc[checkedData.index,"color"] = colorPalette
                    colorData.loc[checkedData.index,"layer"] = np.arange(1,checkedData.index.size+1)

                    if checkedSizes is not None:
                        checkSizeIdx = checkedSizes.index[checkedSizes.index.isin(colorData.index)]
                        colorData.loc[checkSizeIdx,"size"] = checkedSizes.loc[checkSizeIdx]
                    #if user defned color - overwrite
                    if userColors is not None:
                        userColIdx = userColors.index[userColors.index.isin(colorData.index)]
                        colorData.loc[userColIdx,"color"] = userColors.loc[userColIdx]

                    return colorData, None, None

            elif checkedLabels is not None:
               
                nColors = checkedLabels.index.size
                colorPalette = sns.color_palette(colorMap,nColors,desat=self.desat).as_hex()
                idxByCheckedLabel = {}
                #print(checkedLabels)
                for n,idx in enumerate(checkedLabels.index):
                    if idx in userColors.index:
                        color = userColors.loc[idx]
                        #print(color)
                    else:
                        color = rgbToHex(colorPalette[n])
                    
                    boolIdx = self.sourceData.categoricalFilter.searchCategory(dataID,columnName,checkedLabels.loc[idx])
                    
                    idxByCheckedLabel[checkedLabels.loc[idx]] = [idx for idx in boolIdx.index if boolIdx.loc[idx]]
                    
                    colorData.loc[boolIdx,"color"] = color
                    colorData.loc[boolIdx,"layer"] = n+1
                    if checkedSizes is None:
                        colorData.loc[boolIdx,"size"] = self.defaultScatterSize 
                    else:
                        
                        if idx in checkedSizes.index:
                            colorData.loc[boolIdx,"size"] = checkedSizes.loc[idx]
                        else:
                            colorData.loc[boolIdx,"size"] = self.defaultScatterSize 
                
                return colorData, pd.Series(colorPalette,index = checkedLabels.index), idxByCheckedLabel

            return None, None, None

    def createColorMapDict(self,uniqueCategories, nanString = None, addNaNLevels = [], as_hex = False):
        ""

        if nanString is None:
            nanString = self.sourceData.getNaNString()
        #find number of unique categories
        if isinstance(uniqueCategories,list):
            numbUniqueCategories = len(uniqueCategories)
        else:
            numbUniqueCategories = uniqueCategories.size
        if as_hex:
            colors = sns.color_palette(self.colorMap, numbUniqueCategories, desat=self.desat).as_hex()
        else:
            #craete rgb color palette
            colors = sns.color_palette(self.colorMap, numbUniqueCategories, desat=self.desat)
        #create map dict
        colorMapDict = OrderedDict(zip(uniqueCategories ,colors))
        #replace nan with the nan color
        colorMapDict = self.replaceNaNInColorMap(colorMapDict,
                            additionalNanLevels = [nanString] + addNaNLevels)
        layerColorDict = self.buildColorLayerMap(colors)
        return colorMapDict, layerColorDict


    def getCategoricalColorMap(self, dataID, categoricalColumn, userDefinedColors = {}):
        '''
        This is needed for updating the color in scatter plots with same colors
        Retrns a dict of : Categorical level - color

        '''
        naString = self.sourceData.getNaNString()

        if isinstance(categoricalColumn,str):
            categoricalColumn = [categoricalColumn]
        numCategoricalColumn = len(categoricalColumn)
        if  numCategoricalColumn == 1:
            #find unique categories
            uniqueCategories = self.sourceData.getUniqueValues(dataID, categoricalColumn[0])
            colorMapDict, layerColorDict = self.createColorMapDict(uniqueCategories)

        else:
            #find tuple unique categorical values
            combinationsInData = self.sourceData.dfs[dataID][categoricalColumn].apply(tuple,axis=1)
            uniqueCategories = combinationsInData.unique()
            numbUniqueCombData = uniqueCategories.size

            rgbColors = sns.color_palette(self.colorMap,numbUniqueCombData)

            colorMapDict = OrderedDict(zip(uniqueCategories ,rgbColors))

            colorMapDict = self.replaceNaNInColorMap(colorMapDict,
                                additionalNanLevels = [(naString,)*numCategoricalColumn])
            #create a layer :: color mapper. (last color in seaborsn pellet is in front.)
            layerColorDict = self.buildColorLayerMap(rgbColors)

        colorMapDictRaw = colorMapDict.copy()
        for category in uniqueCategories:
                if category in userDefinedColors:
                    colorMapDict[category] = userDefinedColors[category]

       
        colorProps = {"colorMapDict":colorMapDict,
                        "layerColorDict":layerColorDict,
                        "colorMapDictRaw":colorMapDictRaw,
                        "categoricalColumns":categoricalColumn}

        return colorProps

    def getNColorsByCurrentColorMap(self,N,colorMapName = "colorMap"):
        ""
        if hasattr(self,colorMapName):
            cm = getattr(self,colorMapName)
        else:
            cm = colorMapName
        return sns.color_palette(cm,n_colors=N,desat=self.desat).as_hex()
        

    def getColorMap(self):
        ""
        return self.colorMap

    def matchColorsToValues(self,arr = None, colorMapName = None, vmin = None, vmax = None):
        ""
        cmap, colors = self.get_max_colors_from_pallete(colorMapName,returnColors=True)
        if vmin is None or vmax is None:
            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        colorMap = mapper.to_rgba(arr)
        
        return colorMap, colors 
        



    def get_max_colors_from_pallete(self,colorMap = None, returnColors = False):
        '''
        '''
        if colorMap is None:
            colorMap = self.getColorMap()
        #quite ugly. change to dict?
        if colorMap in ["Greys","Blues","Greens","Purples",'Reds',"BuGn","PuBu","PuBuGn","BuPu",
            "OrRd","BrBG","PuOr","Spectral","RdBu","RdYlBu","RdYlGn","viridis","inferno","cubehelix"]:
            n = 20
        elif colorMap in ['Accent','Dark2','Pastel2','Set2','Set4']:
            n=8
        elif colorMap in ['Tenenbaums','Darjeeling Limited','Moonrise Kingdom','Life Acquatic']:
            n = 5
        elif colorMap in ['Paired','Set3']:
            n = 12
        elif colorMap in ['Pastel1','Set1']:
            n = 9
        elif colorMap == 'Set5':
            n = 13
        elif colorMap == 'Set7':
            n = 4
        elif colorMap in self.savedMaxColors:
            n = self.savedMaxColors[colorMap]
        else:
            n = 0
            col = []
            for color in sns.color_palette(colorMap,n_colors=60):
                if color not in col:
                    col.append(color)
                    n += 1
                else:
                    break
            self.savedMaxColors[colorMap] = n
        colors = sns.color_palette(colorMap, n_colors = n ,desat = self.desat)
        cmap = ListedColormap(colors)
        if returnColors:
            return cmap, colors
        else:
            return cmap


    def replaceNaNInColorMap(self,colorMapDict,additionalNanLevels = []):
        ""

        levelsToNaN = ['nan',' '] + additionalNanLevels

        for level in levelsToNaN:
            col_map_update = replaceKeyInDict(level, colorMapDict, self.nanColor)
        return col_map_update


    def setColorMap(self, newColorMap):
        ""
        self.colorMap = newColorMap




    