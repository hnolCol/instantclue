import pandas as pd
import numpy as np
from collections import OrderedDict
from ..utils.stringOperations import getRandomString

def getAxisPostistion(n,nRows = None, nCols = None, maxCol = 4):
    ""
    if nRows is None:
        if n < maxCol:
            nRows = 1
            nCols = n 

        else:
            nRows = np.ceil(n / maxCol)
            nCols = maxCol

    return OrderedDict([(n,[nRows,nCols,n + 1 ]) for n in range(n)])



def calculatePositions(dataID, sourceData, numericColumns, categoricalColumns, maxColumns, **kwargs):
    """
    Return tickposition and axisPositions and colors
    """
    nCatCols = len(categoricalColumns)
    nNumCols = len(numericColumns)
    colorGroups  = pd.DataFrame()
    axisLimits = {}
    axisTitles = {}
    data = sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
    if nCatCols == 0:

        axisPostions = getAxisPostistion(n = 1, maxCol=maxColumns)# dict([(n,[1,1,n+1]) for n in range(1)])
        widthBox = 0.75
        tickPositions = {0:np.arange(nNumCols) + widthBox}
        boxPositions = tickPositions.copy()
        colors,_ = sourceData.colorManager.createColorMapDict(numericColumns, as_hex=True)
        
        colorGroups["color"] = colors.values()
        colorGroups["group"] = colors.keys() 
        colorGroups["internalID"] = [getRandomString() for n in colors.values()]

        faceColors = {0: colors.values()}
        tickLabels = {0:numericColumns}
        filteredData = [data[numericColumn].dropna() for numericColumn in numericColumns]
        groupNames = {0:numericColumns}
        plotData =   {0:{
                        "x":filteredData,
                        }}
        axisLabels = {0:{"x":"","y":"value"}}
        colorCategoricalColumn = "Numeric Columns"

    elif nCatCols == 1:

        axisPostions = getAxisPostistion(n = 1)
        colorCategories = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
        colors,_ = sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
        nColorCats = colorCategories.size
        colorGroups["color"] = colors.values()
        colorGroups["group"] = colorCategories
        colorGroups["internalID"] = [getRandomString() for n in colors.values()]

        filteredData = []
        tickPositions = []
        boxPositions = []
        faceColors = []
        tickLabels = []
        groupNames = []
        widthBox= 1/(nColorCats)
        for m, numericColumn in enumerate(numericColumns):
            numData = data.dropna(subset=[numericColumn])
            startPos = m if m == 0 else m + (widthBox/3 * m)
            endPos = startPos + widthBox* (nColorCats-1)
            positions = np.linspace(startPos,endPos,num=nColorCats)
            tickPos = np.median(positions)
            tickPositions.append(tickPos)
            tickLabels.append(numericColumn)
            for n,(groupName,groupData) in enumerate(numData.groupby(categoricalColumns[0],sort=False)):
                if groupData.index.size > 0:
                    filteredData.append(groupData[numericColumn])
                    faceColors.append(colors[groupName])
                    boxPositions.append(positions[n])
                    groupNames.append("({}:{})::({})".format(categoricalColumns[0],groupName,numericColumn))
                    
        #overriding names, idiot. change!
        tickPositions = {0:tickPositions}
        boxPositions = {0:boxPositions}
        tickLabels = {0:tickLabels}
        faceColors = {0: faceColors}
        groupNames = {0:groupNames}

        plotData =   {0:{
                        "x":filteredData,
                    }}
                       # "capprops":{"linewidth":self.boxplotCapsLineWidth}}}
        axisLabels = {0:{"x":categoricalColumns[0],"y":"value"}}
        colorCategoricalColumn = categoricalColumns[0]
    
    elif nCatCols == 2:

        tickPositions = {}
        faceColors = {}
        boxPositions = {}
        axisLabels = {}
        plotData = {}
        tickLabels = {}
        groupNames = {}
        
        axisPostions = getAxisPostistion(n = len(numericColumns))
        #first category splis data on x axis
        xAxisCategories = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
       # nXAxisCats = xAxisCategories.size
        #second category is color coded
        colorCategories = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
        tickCats = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
        nXAxisCats = tickCats.size

        nColorCats = colorCategories.size
        colors, _  = sourceData.colorManager.createColorMapDict(colorCategories, 
                                                    as_hex=True, 
                                                    )
        widthBox= 1/(nColorCats)
        border = widthBox / 3
        colorGroups["color"] = colors.values()
        colorGroups["group"] = colorCategories
        colorGroups["internalID"] = [getRandomString() for n in colors.values()]
        catGroupby = data.groupby(categoricalColumns, sort=False)

        for nAxis,numColumn in enumerate(numericColumns):
            filteredData = []
            catTickPositions = []
            catBoxPositions = []
            catFaceColors = []
            catTickLabels = []
            catGroupNames = []
            

            for nTickCat, tickCat in enumerate(tickCats):
                startPos = nTickCat if nTickCat == 0 else nTickCat + (border * nTickCat) #add border
                endPos = startPos + widthBox * nColorCats - widthBox
                positions = np.linspace(startPos,endPos,num=nColorCats)
                catTickPositions.append(np.median(positions))
                catTickLabels.append(tickCat)
                for nColCat, colCat in enumerate(colorCategories):
                    groupName = (colCat,tickCat)
                    if groupName not in catGroupby.groups:
                        continue 
                    groupData = catGroupby.get_group(groupName).dropna(subset=[numColumn])
                    if groupData.index.size > 0:
                        filteredData.append(groupData[numColumn])
                        catFaceColors.append(colors[colCat])
                        catBoxPositions.append(positions[nColCat])
                        catGroupNames.append("({}:{}):({}:{})::({})".format(categoricalColumns[1],tickCat,categoricalColumns[0],groupName,numColumn))

            tickPositions[nAxis] = catTickPositions
            faceColors[nAxis] = catFaceColors
            boxPositions[nAxis] = catBoxPositions 
            tickLabels[nAxis] = catTickLabels
            groupNames[nAxis] = catGroupNames
            axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
            plotData[nAxis] =  {
                            "x":filteredData,
                            }
            
            colorCategoricalColumn = categoricalColumns[0]


        # for n,numericColumn in enumerate(numericColumns):
        #     #axis iteration 
            
        #     filteredData = []
        #     catTickPositions = []
        #     catBoxPositions = []
        #     catFaceColors = []
        #     catTickLabels = []
        #     catGroupNames = []
            

        #     numData = data.dropna(subset=[numericColumn])
        #     if not numData.empty:
        #         for m,(axisCat,catData) in enumerate(numData.groupby(categoricalColumns[1],sort=False)):

        #             startPos = m if m == 0 else m + (border * m) #add border
        #             endPos = startPos + widthBox * nColorCats - widthBox
        #             positions = np.linspace(startPos,endPos,num=nColorCats)
        #             tickPos = np.median(positions)
        #             catTickPositions.append(tickPos)
        #             catTickLabels.append(axisCat)
                    
        #             for nColCat,(groupName,groupData) in enumerate(catData.groupby(categoricalColumns[0],sort=False)):
                        
        #                 if groupData.index.size > 0:
                            
                            
                            

            

    elif nCatCols == 3:

        tickPositions = {}
        faceColors = {}
        boxPositions = {}
        axisLabels = {}
        plotData = {}
        tickLabels = {}
        groupNames = {}        

        axisCategories = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[2])
        NNumCol = len(numericColumns)

        axisPostions = getAxisPostistion(n = axisCategories.size *  NNumCol, maxCol = axisCategories.size)
        #print(axisPostions)
        #get color cats
        colorCategories = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
        tickCats = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
        axisCategories = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[2])
        nXAxisCats = tickCats.size
        colorCategoricalColumn = categoricalColumns[0]
        nColorCats = colorCategories.size
        colors, _  = sourceData.colorManager.createColorMapDict(colorCategories, 
                                                    as_hex=True, 
                                                    )
        widthBox= 1/(nColorCats)
        border = widthBox / 3
        colorGroups["color"] = colors.values()
        colorGroups["group"] = colorCategories
        colorGroups["internalID"] = [getRandomString() for n in colors.values()]
        globalMin, globalMax = np.nanquantile(data[numericColumns].values, q = [0,1])
        yMargin = np.sqrt(globalMax**2 + globalMin**2)*0.05
        catGroupby = data.groupby(categoricalColumns,sort=False)
        nAxis = -1

# ###
#          for n,numColumn in enumerate(numericColumns):

#                 for nAxisCat, axisCat in enumerate(axisCategories):
#                     nAxis +=1 
#                     multiScatterKwargs[nAxis] = dict()
#                     interalIDColumnPairs[nAxis] = dict() #nAxis = axis ID
#                     catTickPositions = []
#                     catTickLabels = []
#                     columnNames = []

#                     for nTickCat, tickCat in enumerate(tickCats):

#                         startPos = nTickCat if nTickCat == 0 else nTickCat + (border * nTickCat) #add border
#                         endPos = startPos + widthBox * nColorCats - widthBox
#                         positions = np.linspace(startPos,endPos,num=nColorCats)
#                         tickPos = np.median(positions)
#                         catTickPositions.append(tickPos)
#                         catTickLabels.append(tickCat)

#                         for nColCat, colCat in enumerate(colorCategories):

#                             ###

        for n,numColumn in enumerate(numericColumns):
                

                for nAxisCat, axisCat in enumerate(axisCategories):
                    filteredData = []
                    catTickPositions = []
                    catBoxPositions = []
                    catFaceColors = []
                    catTickLabels = []
                    catGroupNames = []
                    nAxis += 1

                    for nTickCat, tickCat in enumerate(tickCats):
                        startPos = nTickCat if nTickCat == 0 else nTickCat + (border * nTickCat) #add border
                        endPos = startPos + widthBox * nColorCats - widthBox
                        positions = np.linspace(startPos,endPos,num=nColorCats)
                        tickPos = np.median(positions)
                        catTickPositions.append(tickPos)
                        catTickLabels.append(tickCat)

                        for nColCat, colCat in enumerate(colorCategories):
                            groupName = (colCat,tickCat,axisCat)
                            if groupName not in catGroupby.groups:
                                continue

                            groupData = catGroupby.get_group(groupName).dropna(subset=[numColumn])
                            if groupData.index.size  > 0:
                                filteredData.append(groupData[numColumn])
                                catFaceColors.append(colors[colCat])
                                catBoxPositions.append(positions[nColCat])
                                catGroupNames.append("({}:{}):({}:{}):({}:{})::({})".format(categoricalColumns[1],tickCat,categoricalColumns[0],colCat,categoricalColumns[2],axisCat,numColumn))


                    if not nAxisCat in axisTitles:   
                        axisTitles[nAxisCat] = "{}\n{}".format(categoricalColumns[2],axisCat)
                    tickPositions[nAxis] = catTickPositions
                    faceColors[nAxis] = catFaceColors
                    groupNames[nAxis] = catGroupNames
                    boxPositions[nAxis] = catBoxPositions 
                    tickLabels[nAxis] = catTickLabels
                    axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
                    plotData[nAxis] =  {
                                    "x":filteredData,
                                    }
                    axisLimits[nAxis] = {"yLimit":[globalMin-yMargin,globalMax+yMargin],"xLimit":[0-widthBox/2-border,nXAxisCats+widthBox/2+border]}
                    
                    

    
    return plotData, axisPostions, boxPositions, tickPositions, \
            tickLabels, colorGroups, faceColors, colorCategoricalColumn, widthBox, axisLabels, axisLimits, axisTitles, groupNames


#    
#             subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
#             data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
#             colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
#             colors,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)

#             colorGroups  = pd.DataFrame()
#             colorGroups["color"] = colors.values()
#             colorGroups["group"] = colorCategories
            
          