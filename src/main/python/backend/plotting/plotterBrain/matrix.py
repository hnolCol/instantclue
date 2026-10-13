from .helpers import *
from .helpers import _width_functions, _lv_box_ends, _lv_outliers

class MatrixPlotMixin(object):
    """Correlation matrix, heatmap, dendrogram and quadmesh plots."""

    def getCorrmatrixProps(self,dataID,numericColumns, categoricalColumns, figureSize = None, *args,**kwargs):
        ""
        return self.getHeatmapProps(dataID,numericColumns,categoricalColumns, True, figureSize=figureSize,*args,**kwargs)

    def getHeatmapProps(self,dataID, numericColumns, categoricalColumns, corrMatrix = False, groupingName = None,figureSize = None,*args,**kwargs):
        ""
        rowMaxD = None
        colMaxD = None
        rowClustNumber = None
        rowLinkage = None
        ytickLabels = []
        ytickPosition = []
        rowLineCollection = []
        colLineCollection = []
        groupingRectangles = []
        rectangles = []
        clusterColorMap = {}

        #display grouping setting
        displayGrouping = self.sourceData.parent.config.getParam("hclust.display.grouping")
        distanceBetweenClusterMapAndColor = self.sourceData.parent.config.getParam("hclust.color.axis.border.left")
        removeRowsWithNoStd = self.sourceData.parent.config.getParam("hclust.remove.rows.with.no.std")
        colorsByColumnNames = OrderedDict()
        colorsByColumnNamesFiltered = OrderedDict()

        if displayGrouping and groupingName is not None:
            if isinstance(groupingName,str):
                groupingName = [groupingName]

            for gN in groupingName:

                colorsByColumnNames[gN] = self.sourceData.parent.grouping.getColorsForGroupMembers(gN)

                colorsByColumnNamesFiltered[gN] = dict([(k,v) for k,v in colorsByColumnNames[gN].items() if k in numericColumns])
                if len(colorsByColumnNamesFiltered[gN]) == 0: #if no column in grouping - dont display grouping
                   del colorsByColumnNamesFiltered[gN]

        else:
          #  grouping = {}
            displayGrouping = False

        #cluster rows
        rowMetric = self.sourceData.statCenter.rowMetric
        rowMethod = self.sourceData.statCenter.rowMethod
         #cluster columns
        columnMetric = self.sourceData.statCenter.columnMetric
        columnMethod = self.sourceData.statCenter.columnMethod

        try:
            if corrMatrix:

                data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].corr(method = self.corrMatrixMethod)
                if data.dropna().empty:
                    return getMessageProps("Error..","Correlation matrix calculations resulted in a complete NaN matrix.")
            else:
                if rowMetric == "nanEuclidean":
                    nanThreshold = self.sourceData.parent.config.getParam("min.required.valid.values")
                    if nanThreshold > len(numericColumns):
                        nanThreshold = len(numericColumns)
                    data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].dropna(thresh=nanThreshold)
                    #remove no deviation data (Same value)
                    if removeRowsWithNoStd:
                        data = data.loc[data.std(axis=1) != 0,:]
                    rowMetric = "nanEuclidean"
                    if columnMetric != "None":
                        columnMetric = "nanEuclidean"
                else:
                    if (rowMetric != "None" and rowMethod != "None") or  (columnMetric != "None" and columnMethod != "None"):
                        data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"].dropna()
                        #remove no deviation data (Same value)
                        if removeRowsWithNoStd:
                            data = data.loc[data.std(axis=1) != 0,:]
                    else:
                        data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
                        #if no clustering applied, we can keep all values (e.g. just display)

            data = data.astype(float)


            #nRows, nCols = data.shape
            rawIndex = data.index
            rowXLimit, rowYLimit, rowLineCollection = None, None, None
            colXLimit, colYLimit, colLineCollection = None, None, None


            axisDict = self.getClusterAxes(numericColumns,
                    numberOfRows = rawIndex.size,
                    figureSize = figureSize,
                    corrMatrix=corrMatrix,
                    rowOn = rowMetric != "None" and rowMethod != "None",
                    columnOn =  columnMethod != "None" and columnMetric != "None",
                    numberOfGroups=len(colorsByColumnNamesFiltered),
                    displayGrouping = displayGrouping,
                    distanceBetweenClusterMapAndColor = distanceBetweenClusterMapAndColor
                    )

            #perform clustering on rows.
            if data.shape[0] > 1 and rowMetric != "None" and rowMethod != "None":
                rowLinkage, rowMaxD = self.sourceData.statCenter.clusterData(data,rowMetric,rowMethod)

                Z_row = sch.dendrogram(rowLinkage, orientation='left', color_threshold= rowMaxD,
                                    leaf_rotation=90, ax = None, no_plot=True)

                rowXLimit, rowYLimit, rowLineCollection = self.addDendrogram(Z_row,True)

                rowClustNumber = self.getClusterNumber(rowLinkage,rowMaxD)
                ytickPosition, ytickLabels, rectangles, clusterColorMap = self.getClusterRectangles(data,rowMaxD,rowClustNumber,Z_row)
                data = data.iloc[Z_row['leaves']]

            else:
                del axisDict["axRowDendro"]
                Z_row = None

            #perform clustering on columns
            if data.shape[1] > 1 and columnMetric != "None" and columnMethod != "None":
                columnLinkage, colMaxD = self.sourceData.statCenter.clusterData(np.transpose(data.values),columnMetric,columnMethod)

                Z_col = sch.dendrogram(columnLinkage, orientation='top', color_threshold = colMaxD,
                                    leaf_rotation=90, ax = None, no_plot=True)

                data = data.iloc[:,Z_col['leaves']]
                numericColumns = [numericColumns[idx] for idx in Z_col['leaves']]

                colXLimit, colYLimit, colLineCollection = self.addDendrogram(Z_col,False)

            else:
                del axisDict["axColumnDendro"]

            #add grouping rectangles. TO DO: Better just save the props?
            groupingRectangles = []
            if groupingName is not None and displayGrouping and len(colorsByColumnNamesFiltered) > 0:
                for ii, (gN, colorsByColumnNamesForGn) in enumerate(colorsByColumnNamesFiltered.items()):
                    groupingGNRectangles = [
                        Rectangle(
                            xy = (0+10*n,-0.5+ii),
                            width = 10,
                            height = 1,
                            facecolor =  colorsByColumnNamesForGn[columnName] if columnName in  colorsByColumnNamesForGn else self.sourceData.colorManager.nanColor,
                            alpha = 0.75) for n,columnName in enumerate(data.columns.values)]
                    groupingRectangles.extend(groupingGNRectangles)
        except Exception as e:
            print(e)
            return {}

        groupingAxLabels = {"tickLabels":groupingName,"tickPosition":np.arange(len(groupingName))} if groupingName is not None else {}
        return {"newPlot":True,
            "data":{"plotData":data,
                "rowMaxD" : rowMaxD,
                "dataID" : dataID,
                "rawIndex": rawIndex,
                "rowClustNumber" : rowClustNumber,
                "clusterColorMap" : clusterColorMap,
                "rowLinkage" : rowLinkage,
                "Z_row": Z_row,
                "dendrograms":{
                    "row":rowLineCollection,
                    "col":colLineCollection
                },
                "axisLimits":{
                    "rowDendrogram":{"x":rowXLimit,"y":rowYLimit},
                    "columnDendrogram":{"x":colXLimit,"y":colYLimit},
                    "axColumnGrouping":{"x":[0,len(numericColumns)*10],"y":[-0.5,len(colorsByColumnNamesFiltered)-0.5]}
                    },
                "tickLabels" : {"rowDendrogram":{"tickLabels": ytickLabels,"tickPosition": ytickPosition},
                                "axColumnGrouping":groupingAxLabels
                                },
                "absoluteAxisPositions" : axisDict,
                "clusterRectangles": rectangles,
                "groupingRectangles" : groupingRectangles,
                "dataID":dataID,
                "columnNames":numericColumns,
                "params" : [
                    ("Type","Hclust" if not corrMatrix else "Corrmarix"),
                    ("rowMetric",rowMetric),
                    ("rowMethod",rowMethod),
                    ("columnMetric",columnMetric),
                    ("columnMethod",columnMethod),
                    ("nanFilter","all NaNs filtered" if rowMetric != "nanEuclidean" else "minValidValues = {}".format(nanThreshold))]
                    }
                }

    def addDendrogram(self,dendrogram,rotate,*args,**kwargs):
        '''
        Idea is from the seaborn package.
        '''
        line_kwargs = dict(linewidths=.4, colors='k')
        dependent_coord = dendrogram['dcoord']
        independent_coord = dendrogram['icoord']
        max_dependent_coord = max(map(max, dependent_coord))

        if rotate:
            lines = dict(segments = [list(zip(x, y))
                                            for x, y in zip(dependent_coord,
                                            independent_coord)],
                                            **line_kwargs)
            yLimit = len(dendrogram['leaves']) * 10
            xLimit = max_dependent_coord * 1.05
            #ax.set_ylim(0, self.yLimitRow)
            #ax.set_xlim(0, self.xLimitRow)
            #ax.invert_xaxis()

        else:
            lines = dict(segments = [list(zip(x, y))
                                            for x, y in zip(independent_coord,
                                            dependent_coord)],
                                            **line_kwargs)
            xLimit =  len(dendrogram['leaves']) * 10
            yLimit =  max_dependent_coord * 1.05
            #ax.set_xlim(0, self.xLimitCol)
            #ax.set_ylim(0, self.yLimitCol)

        return (0,xLimit), (0,yLimit), lines

    def getClusterRectangles(self,data,rowMaxD,rowClustNumber,Z_row):
        ""
        annotationDf = pd.DataFrame(rowClustNumber,columns=['labels'],index = data.index)
        sortedDataIndex =  data.index[Z_row['leaves']]
        uniqueCluster = annotationDf.loc[sortedDataIndex]['labels'].unique()
        valueCounts = annotationDf.loc[sortedDataIndex]['labels'].value_counts(sort=False)
        countsClust = [valueCounts.loc[clustLabel] for clustLabel in uniqueCluster]
        ytickPosition = [(sum(countsClust[0:n+1])-valueCounts.loc[x]/2)*10 for n,x in enumerate(uniqueCluster)]
        ytickLabels = ['C ({})'.format(cluster) for cluster in uniqueCluster]
        clusterColors = self.sourceData.colorManager.getNColorsByCurrentColorMap(uniqueCluster.size,"hclustClusterColorMap")
        clusterColorMap = dict([(k,v) for k,v in zip(ytickLabels,clusterColors)])
        #clusterColors = sns.color_palette(self.clusterColorMap,uniqueCluster.size)
        rectangles = [dict(
                        xy = (0,n if n == 0 else sum(countsClust[:n] * 10)),
                        width = rowMaxD,
                        height = yLimit * 10,
                        alpha = 0.75,
                        facecolor = clusterColors[n]) for n, yLimit in enumerate(countsClust)]

        return ytickPosition, ytickLabels, rectangles, clusterColorMap

    def getClusterNumber(self,linkage,maxD):
        '''
        Returns cluster numbers
        '''
        return sch.fcluster(linkage,maxD,'distance')

    def getClusterAxes(self,
                    numericColumns,
                    numberOfRows,
                    figureSize,
                    corrMatrix=False,
                    rowOn = True,
                    columnOn = True,
                    numberOfGroups = 0,
                    displayGrouping = False,
                    distanceBetweenClusterMapAndColor = 25):
        ""
        pixelPerColumn = self.sourceData.parent.config.getParam("pixel.width.per.column")
        squareRectangles = self.sourceData.parent.config.getParam("pixel.height.equals.width")
        pixelBorderBottom = self.sourceData.parent.config.getParam("hclust.pixel.border.bottom")
        pixelBorderTop = self.sourceData.parent.config.getParam("hclust.pixel.border.top")
        pixelBorderLeft = self.sourceData.parent.config.getParam("hclust.pixel.border.left")
        pixelFigureWidth = figureSize["width"]
        pixelFigureHeight = figureSize["height"]
        x0,y0 = 0,0
        x1,y1 = 1,1
        if pixelBorderBottom < pixelFigureHeight:
            bottomLabelMargin = pixelBorderBottom / pixelFigureHeight
        else:
            bottomLabelMargin = 0.15

        if pixelBorderBottom < pixelFigureHeight:
            topMargin = pixelBorderTop/ pixelFigureHeight
        else:
            topMargin = 0.22

        leftMargin = pixelBorderLeft/pixelFigureWidth
        if leftMargin > 0.5:
            leftMargin = 0.15

        width = x1-x0-leftMargin
        height = y1-y0-bottomLabelMargin-topMargin

        #multWidth = 0.4


        maxPixelForHeatmap = (width - 0.1) * pixelFigureWidth #0.1 = min margin
        maxPixelHeightHeatmap = height * pixelFigureHeight

        if not corrMatrix:
            heightMain = height


        if maxPixelForHeatmap < len(numericColumns) * pixelPerColumn:
            clusterMapWidth = 0.75
            widthInPixel = maxPixelForHeatmap
        else:
            widthInPixel = len(numericColumns) * pixelPerColumn
            clusterMapWidth = 0.75 * widthInPixel/maxPixelForHeatmap

        borderForColorMap = distanceBetweenClusterMapAndColor/pixelFigureWidth

        if corrMatrix:

            if widthInPixel <= maxPixelHeightHeatmap:

               # heightInPixel = widthInPixel
                heightMain = height * widthInPixel/maxPixelHeightHeatmap

            else:
                widthInPixel = maxPixelHeightHeatmap
                clusterMapWidth = 0.75 * widthInPixel/maxPixelForHeatmap
                heightMain = height * widthInPixel/maxPixelHeightHeatmap

        elif squareRectangles:

            pixelRelativeHeight = (widthInPixel/len(numericColumns) * numberOfRows) / (pixelFigureHeight)
            if pixelRelativeHeight < heightMain:
                heightMain = pixelRelativeHeight

          #  y0 = height - heightMain - 0.1# 0.1#(maxPixelHeightHeatmap - heightInPixel) / maxPixelHeightHeatmap
        # if not corrMatrix:
        #     y0 = bottomLabelMargin

        # else:
        y0 = 1 - heightMain - topMargin
        x0 = leftMargin

        rowDendroWidth = width * 0.08 if rowOn else 0
        if columnOn:
            if rowDendroWidth > 0:
                columnDendroHeight = rowDendroWidth
            else:
                columnDendroHeight = width * 0.13
        else:
            columnDendroHeight = 0

        if numberOfGroups > 0 and displayGrouping:

           groupingAxHeight = width * 0.013 * numberOfGroups
           addSpacingForGroupAx = width * 0.013 * 0.75

        else:
            groupingAxHeight = 0
            addSpacingForGroupAx = 0


        #create axis dict.
        axisDict = dict()

        axisDict["axRowDendro"] = [x0,
                                    y0,
                                    rowDendroWidth,
                                    heightMain]

        axisDict["axColumnDendro"] = [x0 + rowDendroWidth,
                                    y0+heightMain + groupingAxHeight + addSpacingForGroupAx,
                                    clusterMapWidth,
                                    columnDendroHeight ]

        if groupingAxHeight > 0 and displayGrouping:

            axisDict["axColumnGrouping"] = [x0 + rowDendroWidth,
                                    y0+heightMain + addSpacingForGroupAx,
                                    clusterMapWidth,
                                    groupingAxHeight]

        axisDict["axClusterMap"] = [x0+rowDendroWidth,
                                    y0,
                                    clusterMapWidth,
                                    heightMain]

        axisDict["axLabelColor"] =  [x0+rowDendroWidth+clusterMapWidth+borderForColorMap, #add margin
                                    y0,
                                    width-clusterMapWidth-rowDendroWidth if clusterMapWidth < 0.75 else 0.1,
                                    heightMain]

        axisDict["axColormap"] =    [x0,
                                    y1-0.15,
                                    0.02,
                                    0.1]

        return axisDict

    def getColorQuadMeshForHeatmap(self,dataID, colorColumn = None, colorColumnType = None):
        ""
        if colorColumn is None:
            colorColumn = self.colorColumn
        if colorColumnType is None:
            colorColumnType = self.colorColumnType
        colorMeshLimits = None
        colorColumnNames = colorColumn.values
        treatIntegersAsCategories = self.sourceData.parent.config.getParam("hclust.treat.integers.as.categories")
        rawData = self.sourceData.getDataByColumnNames(dataID,colorColumn)["fnKwargs"]["data"]

        if colorColumnType == "Categories" or (colorColumnType == "Integer" and treatIntegersAsCategories):
            combineCategoricalValues = self.sourceData.parent.config.getParam("hclust.color.combine.categories")
            sortUniqueValues = self.sourceData.parent.config.getParam("hclust.color.sort.unique.values")
            ascendingOrder = self.sourceData.parent.config.getParam("hclust.color.sort.unique.values.ascending")
            colorMap = self.sourceData.parent.config.getParam("hclust.color.column.categories.colormap")

            if not combineCategoricalValues  and colorColumn.values.size > 1:
                for columnName in colorColumn.values:
                    rawData[columnName] = rawData[columnName].astype(str) + "({})".format(columnName)
                uniqueValuesList = [rawData[columName].unique() for columName in colorColumn.values]
            else:
                uniqueValuesList = self.sourceData.getUniqueValues(dataID,colorColumn.values.tolist(), forceListOutput=True)

            #unique values. add nan value first -> by default light grey
            # #append replaceObjectNan - should be first item!
            #we need pandas unique to preserve ordering
            if sortUniqueValues:
                uniqueValuesAsSeries = pd.Series(np.concatenate(uniqueValuesList)).sort_values(ascending=ascendingOrder)
            else:
                uniqueValuesAsSeries = pd.Series(np.concatenate(uniqueValuesList))

            uniqueValuesTotal = pd.Series([self.sourceData.replaceObjectNan]).append(
                                uniqueValuesAsSeries, ignore_index=True).unique()
            factorMapper = OrderedDict(zip(uniqueValuesTotal,np.arange(uniqueValuesTotal.size)))
            #map color data to data
            colorData = pd.DataFrame(columns = colorColumn, index = rawData.index)
            for columnName in colorColumn.values:
                colorData.loc[rawData.index,columnName] = rawData.loc[rawData.index,columnName].map(factorMapper)

            colorValues = sns.color_palette(colorMap,len(factorMapper)).as_hex()
            colorValues[0] = self.sourceData.colorManager.nanColor

            cmap = ListedColormap(colorValues)

            colorGroupData = pd.DataFrame(columns=["color","group","internalID"])
            colorGroupData["color"] = colorValues
            colorGroupData["group"] = list(factorMapper.keys())
            colorGroupData["internalID"] = [getRandomString() for n in colorGroupData.index]

        else:
            colorMap = self.sourceData.parent.config.getParam("hclust.color.column.numeric.colormap")
            combinedScale = self.sourceData.parent.config.getParam("hclust.color.combine.numeric.values")

            colorData = pd.DataFrame(columns = colorColumn, index = rawData.index)

            minV, q25, median, q75, maxV = np.nanquantile(rawData.values,q = [0,0.25,0.5,0.75,1])
            cmap = self.sourceData.parent.colorManager.get_max_colors_from_pallete(colorMap)
            cmap.set_bad(self.sourceData.parent.config.getParam("nanColor"))
            colorMeshLimits = {"vmin":minV,"vmax":maxV}


            colorData =  pd.DataFrame(rawData.values,
                                    columns=colorColumn,
                                    index=rawData.index)


            legendColors, _ = self.sourceData.colorManager.matchColorsToValues(
                        arr = [maxV,q75,median,q25,minV],
                        colorMapName = colorMap,
                        vmin = minV,
                        vmax = maxV,
                        asHex=True)

            colorLimitValues = legendColors.flatten().tolist() + [self.sourceData.colorManager.nanColor]

            colorGroupData = pd.DataFrame(columns=["color","group"])
            groupNames = ["Max ({})".format(getReadableNumber(maxV)),
                        "75% Quantile ({})".format(getReadableNumber(q75)),
                        "Median ({})".format(getReadableNumber(median)),
                        "25% Quantile ({})".format(getReadableNumber(q25)),
                        "Min ({})".format(getReadableNumber(minV)),
                        "NaN"]
            colorGroupData["color"] = colorLimitValues
            colorGroupData["group"] = groupNames

        #     def getNumericColorMappingGroupNames(self, maxV,q75,median,q25,minV):
        # ""
        # groupNames = ["Max ({})".format(getReadableNumber(maxV)),
        #                     "75% Quantile ({})".format(getReadableNumber(q75)),
        #                     "Median ({})".format(getReadableNumber(median)),
        #                     "25% Quantile ({})".format(getReadableNumber(q25)),
        #                     "Min ({})".format(getReadableNumber(minV))]
        # return groupNames

        return {
            "colorData":colorData,
            "colorGroupData":colorGroupData,
            "cmap":cmap,
            "title":mergeListToString(colorColumnNames,"\n"),
            "isEditable":False,
            "encodedColumnNames" : colorColumn,
            "colorMeshLimits" : colorMeshLimits
            }

    def getSizeQuadMeshForHeatmap(self,dataID,sizeColumn = None,sizeColumnType = None):
        ""
        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,sizeColumn)["fnKwargs"]["data"]

        if sizeColumnType == "Numeric Floats":

            quantiles = np.nanquantile(rawData.values, q = np.linspace(0,1,num=20))

            repData = np.repeat(rawData.values,quantiles.size,axis=1)
            repData[repData < quantiles] = np.nan
            sizeData = pd.DataFrame(repData,index=rawData.index)

        return {"sizeData":sizeData}

    def getColorGroupingForCorrmatrix(self, columnNames, grouping, groupingName, groupColorMapper):

        #turn values and keys around
        matchDict = {k: oldk for oldk, oldv in grouping.items() for k in oldv}
        groupNames = pd.Series(columnNames.map(matchDict))

        groupFactors, levels = pd.factorize(groupNames)
        groupMapper = dict([(levels[factor],factor) for factor in groupFactors if factor >= 0])

        colorData = pd.DataFrame(groupFactors, columns=["Grouping"], index = columnNames.values)
        colorData["Grouping"] = colorData["Grouping"].replace(-1,np.nan)

        title = groupingName

        colorGroupData = pd.DataFrame(columns=["color","group","internalID"])
        colorGroupData["color"] = [to_hex(groupColorMapper.to_rgba(x)) for x in groupMapper.values()] + [self.sourceData.colorManager.nanColor]
        colorGroupData["group"] = list(groupMapper.keys())  + [self.sourceData.replaceObjectNan]
        colorGroupData["internalID"] = [getRandomString() for n in colorGroupData.index + 1]

        return {"colorData":colorData,"colorGroupData":colorGroupData,"cmap":groupColorMapper,"title":title}
