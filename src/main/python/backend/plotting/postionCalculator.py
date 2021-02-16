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



def calculatePositions(dataID, sourceData, numericColumns, categoricalColumns, maxColumns, splitByCategories = False, **kwargs):
    """
    Return tickposition and axisPositions and colors
    """
    nCatCols = len(categoricalColumns)
    nNumCols = len(numericColumns)
    colorGroups  = pd.DataFrame()
    axisLimits = {}
    axisTitles = {}
    verticalLines = {}
    data = sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]

    if not splitByCategories and nCatCols > 0:

        axisPostions = getAxisPostistion(n = nNumCols, maxCol=2)

        uniqueValueIndex = {}
        tickPositionByUniqueValue = {}
        tickPositions = {}
        boxPositions = {} 
        faceColors = {}
        groupNames = {}
        plotData = {}
        #get unique values

        replaceObjectNan = sourceData.replaceObjectNan

        uniqueValuesByCatColumns = OrderedDict([(categoricalColumn,[cat for cat in data[categoricalColumn].unique() if cat != replaceObjectNan]) for categoricalColumn in categoricalColumns])
        uniqueValuesForCatColumns = [uniqueValuesByCatColumns[categoricalColumn]  for categoricalColumn in categoricalColumns] 
        uniqueValuesPerCatColumn = dict([(categoricalColumn,uniqueValuesForCatColumns[n]) for n,categoricalColumn in enumerate(categoricalColumns)])
        #numUniqueValuesPerCatColumn = dict([(categoricalColumn,len(uniqueValuesForCatColumns[n])) for n,categoricalColumn in enumerate(categoricalColumns)])
        
        uniqueCategories = ["Complete"] + ["{}:({})".format(uniqueValue,k) for k,v in uniqueValuesByCatColumns.items() for uniqueValue in v]
        colors,_ = sourceData.colorManager.createColorMapDict(uniqueCategories, as_hex=True)
        flatUniqueValues = ["Complete"] + [uniqueValue for sublist in uniqueValuesForCatColumns for uniqueValue in sublist]
        #drop "-"
        totalNumUniqueValues = np.array(uniqueValuesForCatColumns).flatten().size
        
        widthBox = 1/(totalNumUniqueValues + 1)
        border = widthBox/10
       
        colorGroups["color"] = colors.values()
        colorGroups["group"] = uniqueCategories
        colorGroups["internalID"] = [getRandomString() for n in colorGroups["color"].values]
        
        colorCategoricalColumn = "\n".join(categoricalColumns)
        #get data bool index

        
        offset = 0 + border + widthBox/2
        tickPositionByUniqueValue["Complete"] = offset

        for categoricalColumn, uniqueValues in uniqueValuesPerCatColumn.items(): 
            uniqueValueIndex[categoricalColumn] = {}
            offset += widthBox/2 #extra offset by categorical column
            
            for uniqueValue in uniqueValues:
                offset += widthBox
                idxBool = data[categoricalColumn] == uniqueValue
                uniqueValueIndex[categoricalColumn][uniqueValue] = idxBool
                tickPositionByUniqueValue["{}:({})".format(uniqueValue,categoricalColumn)] = offset
                # if not uniqueValue == uniqueValues[-1]:
                #     offset += widthBox/5
        offset += border + widthBox/2
        
        for n,numericColumn in enumerate(numericColumns):
            #init lists to stare props
            filteredData = []
            verticalLines[n] = []
            boxPositions[n] = []
            tickPositions[n] = []
            faceColors[n] = []
            groupNames[n] = []
            # add complete data
            filteredData.append(data[numericColumn].dropna())
            boxPositions[n].append(tickPositionByUniqueValue["Complete"] )
            tickPositions[n].append(tickPositionByUniqueValue["Complete"] )
            faceColors[n].append(colors["Complete"])
            groupNames[n].append("{}:Complete".format(numericColumn))
            #iterate through unique values
            for categoricalColumn, uniqueValues in uniqueValuesPerCatColumn.items():
                
                for m,uniqueValue in enumerate(uniqueValues):
                    
                    colorKey ="{}:({})".format(uniqueValue,categoricalColumn)
                    fc = colors[colorKey]
                    idxBool = uniqueValueIndex[categoricalColumn][uniqueValue]
                    uniqueValueFilteredData = data[numericColumn].loc[idxBool].dropna() 
                    tickBoxPos = tickPositionByUniqueValue[colorKey]
                    if m == 0:
                        verticalLines[n].append({
                            "label":categoricalColumn,
                            "color": "darkgrey",
                            "linewidth" : 0.5,
                            "x":tickBoxPos - widthBox/2 - widthBox/4})
                    if uniqueValueFilteredData.index.size > 0:
                        filteredData.append(uniqueValueFilteredData)
                        
                        boxPositions[n].append(tickBoxPos)
                        tickPositions[n].append(tickBoxPos)
                        faceColors[n].append(fc)
                        groupNames[n].append(colorKey)

            plotData[n] = {"x":filteredData}

       
        tickLabels = dict([(n,flatUniqueValues) for n in range(nNumCols)])
        axisLabels = dict([(n,{"x":"Categories","y":numericColumn}) for n,numericColumn in enumerate(numericColumns)])
        axisLimits = dict([(n,{"xLimit" : (0,offset),"yLimit":None}) for n in range(nNumCols)])
    
    elif nCatCols == 0:

        axisPostions = getAxisPostistion(n = 1, maxCol=maxColumns)# dict([(n,[1,1,n+1]) for n in range(1)])
        widthBox = 0.75
        tickValues = np.arange(nNumCols) + widthBox
        tickPositions = {0:tickValues}
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
        axisLimits[0] = {"xLimit" :  (tickValues[0]- widthBox,tickValues[-1] + widthBox), "yLimit" : None} 

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
            endPos = startPos + widthBox * (nColorCats-1)
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
                    
        axisLimits[0] = {"xLimit" :  (boxPositions[0] - widthBox, boxPositions[-1] + widthBox), "yLimit" : None} 
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
        #xAxisCategories = sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
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
                        axisTitles[nAxisCat] = "{}:{}".format(categoricalColumns[2],axisCat)
                    tickPositions[nAxis] = catTickPositions
                    faceColors[nAxis] = catFaceColors
                    groupNames[nAxis] = catGroupNames
                    boxPositions[nAxis] = catBoxPositions 
                    tickLabels[nAxis] = catTickLabels
                    axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
                    plotData[nAxis] =  {
                                    "x":filteredData,
                                    }
                    axisLimits[nAxis] = {
                        "yLimit":(globalMin-yMargin,globalMax+yMargin),
                        "xLimit":(catBoxPositions[0] - widthBox, catBoxPositions[-1] + widthBox)
                        }
                    
                        
    return plotData, axisPostions, boxPositions, tickPositions, \
            tickLabels, colorGroups, faceColors, colorCategoricalColumn, widthBox, axisLabels, axisLimits, axisTitles, groupNames, verticalLines


          