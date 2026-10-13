from .helpers import *
from .helpers import _width_functions, _lv_box_ends, _lv_outliers

class ScatterPlotMixin(object):
    """Scatter, dimensionality reduction, PCA, forest, regression and wordcloud plots."""

    def getDimRedProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        return {"data":{}}

    def getForestplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        tickLabels = {}
        tickPositions = {}
        axisLabels = {}
        axisLimits = {}

        axisLabelConverter = {
                            "oddsRatio":"Odds ratio",
                            "logOddsRatio":"log (Odds ratio)",
                            "riskRatio": "Risk ratio"
                            }

        if not (len(numericColumns) > 0 and len(categoricalColumns) > 0):
            return getMessageProps("Error ..","")
        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        plotData = OrderedDict()

        faceColors = dict()
        useColorMap = self.sourceData.parent.config.getParam("forest.plot.use.colormap")
        defaultColor = self.sourceData.parent.config.getParam("forest.plot.marker.color")
        colorGroupsData = pd.DataFrame()
        internalIDs = []

        if  self.sourceData.parent.config.getParam("forest.plot.calculated.data"):
            axisPositions = getAxisPosition(1)
            #categorical columns -> names of variables -> yticks
            categoricalData = data[categoricalColumns[0]].values.flatten()
            colors, _ = self.sourceData.colorManager.createColorMapDict(categoricalData, as_hex=True)

            if useColorMap:
                colorGroupsData["color"] = colors.values()
            else:
                colorGroupsData["color"] = [defaultColor] * categoricalData.size
            colorGroupsData["group"] = categoricalData
            tickLabels[0] = categoricalData
            tickPositions[0] = np.arange(categoricalData.size)
            axisLimits[0] = {"yLimit":[-0.5,categoricalData.size-0.5],"xLimit": None}
            axisLabels[0] = {
                        "x":numericColumns[0],
                        "y":"Variables"
                        }

            faceColors[0] = {}
            if len(numericColumns) == 3:
                    meanColumName,ci_lowerColumName, ci_upperColumName = numericColumns
            else:
                    meanColumName = numericColumns[0]
                    ci_lowerColumName = None
                    ci_upperColumName = None

            plotData[0] = OrderedDict()
            for m,idx in enumerate(data.index):

                meanValue = data.loc[idx,meanColumName]
                rowName = categoricalData[m]
                ci_lower = data.loc[idx,ci_lowerColumName]  if ci_lowerColumName is not None else np.nan
                ci_upper = data.loc[idx,ci_upperColumName] if ci_upperColumName is not None else np.nan
                internalID = getRandomString()
                plotData[0][rowName] = {
                            "ratio"  :   meanValue,
                            "ratioCI":   (ci_lower, ci_upper),
                            "yPosition":  m,
                            "internalID" : internalID
                        }
                internalIDs.append(internalID)
                if not useColorMap:
                    faceColors[0][rowName] = defaultColor
                else:
                    faceColors[0][rowName] = colors[rowName]
        else:

            axisPositions = getAxisPosition(len(numericColumns))

            colors, layer = self.sourceData.colorManager.createColorMapDict(categoricalColumns, as_hex=True)
            if useColorMap:
                colorGroupsData["color"] = colors.values()
            else:
                colorGroupsData["color"] = [defaultColor] * len(categoricalColumns)
            colorGroupsData["group"] = categoricalColumns
            for n,numericColumn in enumerate(numericColumns):
                plotData[n] = OrderedDict()
                for m,categoricalColumn in enumerate(categoricalColumns):

                    ct=pd.crosstab(data[numericColumn],data[categoricalColumn])
                    table = Table2x2(ct.values)
                    internalID = getRandomString()

                    plotData[n][categoricalColumn] = {
                                        "oddsRatio"          :   table.oddsratio,
                                        "oddsRatioCI"        :   table.oddsratio_confint(),
                                        "oddsRatioPValue"    :   table.oddsratio_pvalue(),
                                        "logOddsRatio"       :   table.log_oddsratio,
                                        "logOddsRatioCI"     :   table.log_oddsratio_confint(),
                                        "logOddsRatioP"      :   table.log_oddsratio_pvalue(),
                                        "riskRatio"          :   table.riskratio,
                                        "riskRatioCI"        :   table.riskratio_confint(),
                                        "yPosition"          :   m,
                                        "internalID"         :   internalID
                                        }
                    internalIDs.append(internalID)
                tickLabels[n] = categoricalColumns
                axisLabels[n] = {
                        "x":axisLabelConverter[self.sourceData.parent.config.getParam("forest.plot.cont.table.ratio")],
                        "y":"Variables"
                        }
                tickPositions[n] = np.arange(len(categoricalColumns))
                axisLimits[n] = {"yLimit":[-0.5,len(categoricalColumns)-0.5],"xLimit": None}

                if not useColorMap:
                    faceColors[n][categoricalColumn] = defaultColor
                else:
                    faceColors[n][categoricalColumn] = colors[rowName]

        colorGroupsData["internalID"] = internalIDs



        return {"data":{
                    "plotData" : plotData,
                    "axisPositions" : axisPositions,
                    "tickLabels"    : tickLabels,
                    "axisLabels"    : axisLabels,
                    "tickPositions" : tickPositions,
                    "axisLimits"    : axisLimits,
                    "faceColors"    : faceColors,
                    "dataColorGroups": colorGroupsData
                    }
                }

    def getPCAProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        subplotBorders = dict(wspace=0.30, hspace = 0.30,bottom=0.15,right=0.95,top=0.95)
        #data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
        checkPassed, driverResult, eigVectors  = self.sourceData.statCenter.runPCA(dataID,numericColumns, initGraph = True, n_components = 3)
        if not checkPassed:
            return getMessageProps("Error ..","Filtering resulted in an invalid data frame.")

        columnPairs = [("Component_01","Component_02"), ("Component_02","Component_03")]
        #print(result)
      #  print(eigVectors)
        axisPostions = dict([(n,[2,2,n+1]) for n in range(4)])


        return {"data":{"plotData":{"projection":driverResult,"eigV":eigVectors},
                "axisPositions":axisPostions,
                "numericColumns":numericColumns,
                "subplotBorders":subplotBorders,
                "columnPairs":columnPairs,
                "dataID":dataID}}

    def _addIndexToData(self,data,numericColumns):
        ""
        numericColumnPairs = [("Index ({:02d})".format(n+1),numColumn) for n,numColumn in enumerate(numericColumns)]
        for indexName, numColumn in numericColumnPairs:

            index = data.sort_values(by=numColumn, ascending = self.indexSort == "ascending").index
            data = data.join(pd.Series(np.arange(index.size),index=index, name = indexName))

        return data,numericColumnPairs

    def getScatterProps(self,dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""

        try:
            axisTitles = {}

            if len(categoricalColumns) == 0:
                data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]

                if not self.plotAgainstIndex and len(numericColumns) > 1:
                    numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))

                else:
                    data, numericColumnPairs = self._addIndexToData(data,numericColumns)

                nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
                axisPositions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
                axisLabels = dict([(n,{"x":x1,"y":x2}) for n, (x1,x2) in enumerate(numericColumnPairs)])

            else:

                subplotBorders = dict(wspace=0, hspace = 0.2, bottom=0.15,right=0.95,top=0.95)
                data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
                uniqueValuesCat1 = data[categoricalColumns[0]].unique()
                numUniqueCat = uniqueValuesCat1.size


                axisLabels = {}
                firstAxisRow = True
                axisID = 0
                plotNumericPairs = []

                if not self.plotAgainstIndex and len(numericColumns) > 1:
                    numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))
                else:
                    data, numericColumnPairs = self._addIndexToData(data,numericColumns)

                if len(categoricalColumns) == 1:
                    numOfAxis = len(numericColumnPairs) * numUniqueCat
                    axisPositions = getAxisPosition(numOfAxis,maxCol=numUniqueCat)
                    requiredColumns = []
                    for numCols in numericColumnPairs:
                        for uniqueValue in uniqueValuesCat1:
                            requiredColumns.append("{}:{}:({})".format(uniqueValue,categoricalColumns[0],numCols[0]))
                            requiredColumns.append("{}:{}:({})".format(uniqueValue,categoricalColumns[0],numCols[1]))

                    plotData = pd.DataFrame(
                                index=data.index,
                                columns = requiredColumns
                                )


                    for numCols in numericColumnPairs:
                        for nGroup , (groupName, groupData) in enumerate(data.groupby(categoricalColumns[0],sort=False)):

                            if axisID == numUniqueCat:
                                firstAxisRow = False
                            elif firstAxisRow:
                                axisTitles[axisID] = {"title":"{}: {}".format(categoricalColumns[0],groupName),
                                                                "appendWhere":"top",
                                                                "textRotation" : 0}
                            pair = []
                            for columnName in numCols:
                                columnKey = "{}:{}:({})".format(groupName,categoricalColumns[0],columnName)
                                plotData.loc[groupData.index,columnKey] = groupData[columnName]
                                pair.append(columnKey)
                            plotNumericPairs.append(tuple(pair))

                            axisLabels[axisID] = {"x":numCols[0],"y":numCols[1] if nGroup == 0 else ""}
                            axisID += 1

                   # axisPositions = getAxisPosition(len(numericColumnPairs),nCols=numUniqueCat)
                elif len(categoricalColumns) == 2:
                    uniqueValuesCat2 = data[categoricalColumns[1]].unique()
                    numUniqueCat2 = uniqueValuesCat2.size

                    numOfAxis = numUniqueCat * numUniqueCat2 * len(numericColumnPairs)
                    axisPositions = getAxisPosition(numOfAxis,maxCol=numUniqueCat)
                    requiredColumns = []
                    for numCols in numericColumnPairs:
                        for uniqueValueCat2 in uniqueValuesCat2:
                            for uniqueValueCat1 in uniqueValuesCat1:

                                requiredColumns.append("{}:{}:{}:({})".format(uniqueValueCat1,
                                                                            uniqueValueCat2,
                                                                            categoricalColumns[0],
                                                                            numCols[0]))
                                requiredColumns.append("{}:{}:{}:({})".format(uniqueValueCat1,
                                                                            uniqueValueCat2,
                                                                            categoricalColumns[0],
                                                                            numCols[1]))

                    plotData = pd.DataFrame(
                                index=data.index,
                                columns = requiredColumns
                                )


                    for numCols in numericColumnPairs:
                        for uniqueValueCat2, cat2data in data.groupby(categoricalColumns[1],sort=False):

                            for nGroup,  (groupName, groupData) in enumerate(cat2data.groupby(categoricalColumns[0], sort=False)):
                                    if axisID == numUniqueCat:
                                        firstAxisRow = False
                                    elif firstAxisRow:
                                        axisTitles[axisID] = {"title":"{}:{}".format(categoricalColumns[0],groupName),
                                                                "appendWhere":"top",
                                                                "textRotation" : 0}
                                    if groupName == uniqueValueCat1[-1]:
                                        titleProps = {
                                                        "title":"{}:{}".format(categoricalColumns[1],uniqueValueCat2),
                                                        "appendWhere":"right",
                                                        "textRotation" : 90
                                                    }

                                        if axisID in axisTitles:
                                            addTitles = [axisTitles[axisID],titleProps]
                                            axisTitles[axisID] = addTitles
                                        else:
                                            axisTitles[axisID] = titleProps

                                    pair = []
                                    for columnName in numCols:
                                        columnKey = "{}:{}:{}:({})".format(groupName,uniqueValueCat2,categoricalColumns[0],columnName)
                                        plotData.loc[groupData.index,columnKey] = groupData[columnName]
                                        pair.append(columnKey)
                                    plotNumericPairs.append(tuple(pair))

                                    axisLabels[axisID] = {"x":numCols[0],"y":numCols[1] if nGroup == 0 else ""}
                                    axisID += 1
                else:
                    return getMessageProps("Error ..","For Scatter plots only two categorical columns are considered. You can use color, size and marker highlights.")

                numericColumnPairs = plotNumericPairs
                data = plotData




        except Exception as e:
                print(e)

        colorInternalID = getRandomString()
        colorGroupsData = pd.DataFrame()
        colorGroupsData["color"] = [self.sourceData.colorManager.nanColor]
        colorGroupsData["group"] = [""]
        colorGroupsData["internalID"] = [colorInternalID]
        sizeInternalID = getRandomString()
        sizeGroupsData = pd.DataFrame()
        sizeGroupsData["size"] = [self.scatterSize]
        sizeGroupsData["group"] = [""]
        sizeGroupsData["internalID"] = [sizeInternalID]

        colorCategoryIndexMatch = {colorInternalID : data.index}
        sizeCategoryIndexMatch = {sizeInternalID : data.index}

        return {"data":{
            "plotData":data,
            "axisPositions":axisPositions,
            "colorCategoryIndexMatch" : colorCategoryIndexMatch,
            "sizeCategoryIndexMatch" : sizeCategoryIndexMatch,
            "axisTitles": axisTitles,
            "columnPairs":numericColumnPairs,
            "dataColorGroups": colorGroupsData,
            "dataSizeGroups" : sizeGroupsData,
            "axisLabels" : axisLabels,
            "subplotBorders":subplotBorders,
            "dataID":dataID}}

    def getMarkerGroupsForScatter(self,dataID,markerColumn,markerColumnType):
        ""
        categoryIndexMatch = None
        markerGroupData = pd.DataFrame(columns=["marker","group","internalID"])
        markerColumnName = markerColumn.values[0]
        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,markerColumn)["fnKwargs"]["data"]
        if markerColumnType == "Integers":
            markerColumnType, rawData = self._checkIntegerColumn(markerColumn,rawData)

        if markerColumnType == "Categories":
            if markerColumn.index.size > 1:
                rawMarkerData = rawData.apply(tuple,axis=1)
            else:
                rawMarkerData = rawData[markerColumnName]
            markerCategories = rawMarkerData.unique()

            availableMarkers = ["o","^","D","s","P","v","p","+","."]

            markerMap = OrderedDict([(cat,availableMarkers[n % len(availableMarkers)]) for n,cat in enumerate(markerCategories)])

            markerData = rawMarkerData.map(markerMap)
            markerGroupData["marker"] = markerMap.values()
            markerGroupData["group"] = markerCategories
            markerGroupData["internalID"] = [getRandomString() for n in markerGroupData.index]

        propsData = pd.DataFrame(markerData.values,columns=["marker"], index=rawData.index)
        title = mergeListToString(markerColumn.values,"\n")

        categoryIndexMatch = dict([(intID,rawData[rawData.values == category].index) for category, intID in zip(markerGroupData["group"].values,
                                                                                                                markerGroupData["internalID"].values)])

        return {"propsData":propsData,
                "title":title,
                "markerGroupData":markerGroupData,
                "categoryEncoded":"marker",
                "categoryIndexMatch":categoryIndexMatch}

    def _buildColumnPairs(self,numericColumns):
        ""
        numNumericColumns = len(numericColumns)
        #np.zeros(shape=(numNumericColumns,numNumericColumns))
        columnPairs = {}
        n = 0
        for nRow in range(numNumericColumns):
            for nCol in range(numNumericColumns):
                yValue = numericColumns[nRow]
                xValue = numericColumns[nCol]
                columnPairs[n] = tuple([xValue,yValue])
                n += 1

        return columnPairs

    def _build2DHistogramData(self, axisInts, columnPairs, data, histogramData):
        #get pcolormesh for column pairs
        ""
        for nA in axisInts.flatten():
            numericPairs = list(columnPairs[nA])

            XY  = data[numericPairs].dropna().values

            H, xedges, yedges = np.histogram2d(XY[:,0], XY[:,1], bins=(25, 25))
            meshX, meshY = np.meshgrid(xedges, yedges)
            histogramData[nA] = {
                        "H":H,
                        "xedges":xedges,
                        "yedges":yedges,
                        "meshX":meshX,
                        "meshY":meshY
                        }
        return histogramData

    def _buildScatterPairs(self,axisInts,columnPairs,scatterColumnPairs):
        ""
        for nA in axisInts.flatten():
            scatterColumnPairs[nA] = columnPairs[nA]
        return scatterColumnPairs

    def _findBwByGridSearch(self,x,minBW,maxBW,numCrossValidations=5,kernel="gaussian"):
        ""
        grid = GridSearchCV(KernelDensity(kernel=kernel,algorithm='ball_tree'),
                    {'bandwidth': np.linspace(minBW, maxBW, 30)},
                    cv=numCrossValidations) # 5-fold cross-validation
        grid.fit(x)

        return grid.best_params_["bandwidth"]

    def _buildKdeData(self,axisInts,data,numericColumns,kdeData):
        ""

        minMaxValues = np.nanquantile(data[numericColumns].values, q = [0.0,1.0],axis=0)
        bw = self.sourceData.parent.config.getParam("multi.scatter.kde.bandwidth")
        kernel = self.sourceData.parent.config.getParam("multi.scatter.kde.kernel")
        bwGridSearch = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search")
        logDensity = self.sourceData.parent.config.getParam("multi.scatter.kde.log.density")
        gridMin = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.min")
        gridMax = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.max")
        numCVs = self.sourceData.parent.config.getParam("multi.scatter.kde.grid.n.cross.val")
        for n,nAx in enumerate(axisInts):
            x = data[numericColumns[n]].dropna().values.reshape(-1,1)

            #find x limits
            minValue,maxValue = minMaxValues[:,n]
            offSet = (maxValue - minValue) * 0.1
            minValue -= offSet
            maxValue += offSet

            #caluclate kde and get y values
            xx = np.linspace(minValue,maxValue,num=500).reshape(-1,1)
            if bwGridSearch:

                bw = self._findBwByGridSearch(x,gridMin,gridMax,numCVs)


            kde = KernelDensity(bandwidth=bw, kernel=kernel, algorithm='ball_tree').fit(x)
            y = kde.score_samples(xx)
            if not logDensity:
                y = np.exp(y)

            #find y limits
            yMax = np.max(y)
            yMin = np.min(y)
            yMax += 0.15*np.sqrt((yMax-yMin)**2)
            #add kdea data - key is axis int
            kdeData[nAx] = {"xx" : xx, "x" : x, "yKde" : y ,"xLimit":(minValue,maxValue),"yLimit":(yMin,yMax)}

        return kdeData

    def _buildLabelData(self,axisInts, numericColumns, groupColorDict, labelData):
        ""
        for numericColumn, nAx in zip(numericColumns,axisInts):

            tKwargs = {
                "color" : groupColorDict[numericColumn] if numericColumn in groupColorDict else "black",
                "s" : numericColumn,
                "horizontalalignment" : 'center',
                "verticalalignment" : 'center',
                "x" : 0.5,
                "y" : 0.5
                }

            labelData[nAx] = tKwargs


        return labelData

    def _buildCorrelationLabelData(self,axisInts, columnPairs, labelData, corrMatrix  = None, spearmanCorrMatrix=None):
        ""
        if corrMatrix is None and spearmanCorrMatrix is None:
            return labelData

        for nA in axisInts.flatten():
            colA, colB = columnPairs[nA]
            pearCorr = "" if corrMatrix is None else "r = {}\n".format(round(corrMatrix.loc[colA, colB],2))
            spearCorr = "" if spearmanCorrMatrix is None else "rho = {}".format(round(spearmanCorrMatrix.loc[colA, colB],2))

            tKwargs = {
                "color" : "black",
                "s" : pearCorr+spearCorr,
                "horizontalalignment" : 'left',
                "verticalalignment" : 'top',
                "x" : 0.02,
                "y" : 0.95
                }
            labelData[nA] = tKwargs

        return labelData

    def getScatterCorrMatrix(self,dataID, numericColumns, categoricalColumns,*args,**kwargs):
        """
        Provide required information to plot a scatter matrix.
        Several options are taken from the config settings.
        """
        #init dicts
        histogramData = dict()
        kdeData = dict()
        scatterColumnPairs = dict()
        colorDict = dict()
        groupColorDict = dict()
        labelData = dict()
        lowessFit,linregressFit = dict(), dict()
        backgroundColors = dict()
        backgroundColorHex = None
        categoryIndexMatch = None

        config = self.sourceData.parent.config

        topPlotType = config.getParam("multi.scatter.top.right.type")
        bottomPlotType = config.getParam("multi.scatter.bottom.left.type")
        diagPlotType = config.getParam("multi.scatter.diag.type")

        #add linear regression
        addLinReg = config.getParam("multi.scatter.add.linregress")
        addLowess = config.getParam("multi.scatter.add.lowess")
        lowessFrac = config.getParam("multi.scatter.lowess.frac")
        lowessIt = config.getParam("multi.scatter.lowess.it")
        lowessDelta = config.getParam("multi.scatter.lowess.delta")

        #correlation
        addToPlotType = config.getParam("multi.scatter.add.corr.coeff.to")
        addPearson = config.getParam("multi.scatter.add.pearson")
        addSpearman = config.getParam("multi.scatter.add.spearman")
        colorBackground = config.getParam("multi.scatter.background.by.pearson")
        backgroundCmap = config.getParam("multi.scatter.background.colormap")
        colorScale = config.getParam("multi.scatter.background.colorScale")
        #get data
        columnSorting = config.getParam("multi.scatter.sort.columns.by")

        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        corrmatrix = data[numericColumns].corr()
        spearmanCorrMatrix =  data[numericColumns].corr(method="spearman") if addSpearman else None
        nNumCols = len(numericColumns)

        if colorBackground:
            if colorScale == "dataMinMax":
                vmin, vmax = None, None
            else:
                vmin,vmax = [float(x) for x in colorScale.split(",")]
            backgroundColorMapper = self.sourceData.colorManager.matchColorsToValues(corrmatrix.values,backgroundCmap,vmin,vmax)

            backgroundColorHex = pd.DataFrame(
                    np.array([to_hex(c)  for cs in backgroundColorMapper[0] for c in cs ]).reshape((nNumCols,nNumCols)),
                    columns = numericColumns, index = numericColumns)

        if  columnSorting != "None":
            if columnSorting == "Grouping" and self.sourceData.parent.grouping.groupingExists():
                groupingName = self.sourceData.parent.grouping.getCurrentGroupingName()
                columnNames = self.sourceData.parent.grouping.getColumnNames(groupingName)
                columnsFoundInGrouping = [colName for colName in columnNames if colName in numericColumns]
                numericColumns = columnsFoundInGrouping + [colName for colName in numericColumns if colName not in columnsFoundInGrouping]
                groupColorDict = self.sourceData.parent.grouping.getColorsForGroupMembers(groupingName)
            if columnSorting == "Hierarch. Clustering":
                columnLinkage, colMaxD = self.sourceData.statCenter.clusterData(np.transpose(data[numericColumns].dropna(axis=0).values))
                if columnLinkage is None:
                    return getMessageProps("Error..","There was an error in clustering columns.")
                Z_col = sch.dendrogram(columnLinkage, orientation='top', color_threshold = colMaxD,
                                    leaf_rotation=90, ax = None, no_plot=True)

                data = data.iloc[:,Z_col['leaves']]
                numericColumns = [numericColumns[idx] for idx in Z_col['leaves']]


        nAxis = int(nNumCols ** 2)
        axisPositions = getAxisPosition(nAxis, maxCol=nNumCols)
        subplotBorders = dict(wspace=0.1, hspace = 0.1, bottom=0.15,right=0.95,top=0.95)
        #numericColumnPairs =  list(itertools.product(numericColumns,2))  #list(zip(numericColumns[0::2], numericColumns[1::2]))
        columnPairs = self._buildColumnPairs(numericColumns)

        if len(categoricalColumns) > 0:
            colorDict = self.getColorGroupsDataForScatter(dataID,pd.Series(categoricalColumns))

            colorGroupsData = colorDict["colorGroupData"]
            colorTitle = colorDict["title"]
            propsData = colorDict["propsData"]
            categoryIndexMatch = colorDict["categoryIndexMatch"]
        else:
            colorGroupsData = pd.DataFrame()
            if len(categoricalColumns) == 0:
                colorGroupsData["color"] = [self.sourceData.colorManager.nanColor]
                colorGroupsData["group"] = [""]
            colorTitle = "Scatter points"
            propsData = pd.DataFrame(index = data.index, columns = ["color","layer"])
            propsData["color"] = self.sourceData.colorManager.nanColor
            propsData["layer"] = 0
            # else:
            #     colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #     nColorCats = colorCategories.size
            #     colorDict,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)

            #     colorGroupsData["color"] = list(colorDict.values())
            #     colorGroupsData["group"] = list(colorDict.keys())
            #     colorGroupsData["internalID"] = [getRandomString() for _ in range(nColorCats)]

        axisInts = np.array(np.arange(nAxis)).reshape((nNumCols,nNumCols))
        axisIntsTop = np.triu(axisInts)
        axisIntsBottom = np.tril(axisInts)
        np.fill_diagonal(axisIntsTop,0)
        np.fill_diagonal(axisIntsBottom,0)

        #get ints for scatter by using the indicies found by tril/diagonal is zero
        axisIntsTop = axisIntsTop.flatten()
        axisIntsTop = axisIntsTop[axisIntsTop > 0]

        axisIntsBottom = axisIntsBottom.flatten()
        axisIntsBottom = axisIntsBottom[axisIntsBottom > 0]

        axisIntsDiagonal = np.diag(axisInts)

        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")

        kdeKwargs = {"bw" :  self.sourceData.parent.config.getParam("multi.scatter.kde.bandwidth"),
                    "kernel" :  self.sourceData.parent.config.getParam("multi.scatter.kde.kernel"),
                    "bwGridSearch" :  self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search"),
                    "logDensity" :  self.sourceData.parent.config.getParam("multi.scatter.kde.log.density"),
                    "gridMin" :  self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.min"),
                    "gridMax" :  self.sourceData.parent.config.getParam("multi.scatter.kde.grid.search.max"),
                    "numCVs" : self.sourceData.parent.config.getParam("multi.scatter.kde.grid.n.cross.val")}
        lowessKwargs = {"frac":lowessFrac , "it" : lowessIt, "delta" : lowessDelta}
                            # if addLowess:
                            #     lowessFit[nA] = self.sourceData.statCenter.runLowess(dataID,numericPairs,frac=lowessFrac , it = lowessIt, delta = lowessDelta)
        #multi processing to calculate the axisInts (topRight,bottomleft,and diagonal simult.)
        #
        #

        if nNumCols > 5:
            with Pool(NProcesses) as p:
                rPool = p.starmap(buildScatterMatrix,[(data,
                                            numericColumns,
                                            plotType,
                                            axisInts,
                                            backgroundColorHex,
                                            scatterColumnPairs,
                                            colorBackground,
                                            addLinReg,
                                            addLowess,
                                            columnPairs,
                                            kdeKwargs,
                                            groupColorDict,
                                            addToPlotType,
                                            addPearson,
                                            addSpearman,
                                            corrmatrix,
                                            spearmanCorrMatrix,
                                            lowessKwargs) for plotType, axisInts in zip([topPlotType,bottomPlotType,diagPlotType],
                                                                                                    [axisIntsTop,axisIntsBottom,axisIntsDiagonal])])
           # print(rPool) #output of pool - merged:(scatterColumnPairs,backgroundColors,linregressFit,histogramData,kdeData,labelData,lowessData)
        else:
            rPool = [buildScatterMatrix(data,
                                            numericColumns,
                                            plotType,
                                            axisInts,
                                            backgroundColorHex,
                                            scatterColumnPairs,
                                            colorBackground,
                                            addLinReg,
                                            addLowess,
                                            columnPairs,
                                            kdeKwargs,
                                            groupColorDict,
                                            addToPlotType,
                                            addPearson,
                                            addSpearman,
                                            corrmatrix,
                                            spearmanCorrMatrix,
                                            lowessKwargs) for plotType,axisInts in zip([topPlotType,bottomPlotType,diagPlotType],
                                                                                                    [axisIntsTop,axisIntsBottom,axisIntsDiagonal])]
        scatterColumnPairs = {k: v for d in rPool for k, v in d[0].items()}
        backgroundColors = {k: v for d in rPool for k, v in d[1].items()}
        linregressFit = {k: v for d in rPool for k, v in d[2].items()}
        histogramData = {k: v for d in rPool for k, v in d[3].items()}
        kdeData = {k: v for d in rPool for k, v in d[4].items()}
        labelData = {k: v for d in rPool for k, v in d[5].items()}
        lowessFit = {k: v for d in rPool for k, v in d[6].items()}



        #compile size groups
        sizeGroupsData = pd.DataFrame()
        sizeGroupsData["size"] = [self.scatterSize]
        sizeGroupsData["group"] = [""]

        return {"data":{
                        "plotData":data,
                        "axisPositions":axisPositions,
                        "scatterColumnPairs": scatterColumnPairs,
                        "dataColorGroups": colorGroupsData,
                        "dataSizeGroups" : sizeGroupsData,
                        "histogramData" : histogramData,
                        "kdeData" : kdeData,
                        "subplotBorders":subplotBorders,
                        "dataID":dataID,
                        "linRegFit" : linregressFit,
                        "lowessFit": lowessFit,
                        "propsData" : propsData,
                        "colorTitle" : colorTitle,
                        "labelData" : labelData,
                        "backgroundColors" : backgroundColors,
                        "categoryIndexMatch" : categoryIndexMatch
                        }
            }

    def _checkIntegerColumn(self,columnName,rawData):
        ""
        uniqueValues = np.unique(rawData[columnName].values).size
        if uniqueValues > rawData.index.size * 0.75 / columnName.size:
            columnType = "Numeric Floats"
            rawData = rawData.astype(float)
        else:
            columnType= "Categories"
            rawData = rawData.astype(str)

        return columnType, rawData

    def getNearestNeighborConnections(self,dataID,numericColumnPairs,numberNearestNeighbors=3):
        ""

        columnNames = pd.Series(np.array(numericColumnPairs).flatten()).unique()

        rawData = self.sourceData.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]

        lineCollections = {}
        for n,columnPair in enumerate(numericColumnPairs):
            X = rawData[list(columnPair)].dropna().values

            euclideanDistanceMatrix = squareform(pdist(X, 'euclidean'))

            # select the kNN for each datapoint
            neighbors = np.sort(np.argsort(euclideanDistanceMatrix, axis=1)[:, 0:numberNearestNeighbors])

            N = neighbors.shape[0]
            coordinates = np.zeros((N, numberNearestNeighbors, 2, 2))
            for i in np.arange(N):
                for j in np.arange(numberNearestNeighbors):
                    coordinates[i, j, :, 0] = np.array([X[i,:][0], X[neighbors[i, j], :][0]])
                    coordinates[i, j, :, 1] = np.array([X[i,:][1], X[neighbors[i, j], :][1]])
            lineCollections[n] = {"segments":coordinates.reshape((N*numberNearestNeighbors, 2, 2)), "color":'black', "linewidth" : 0.5,"zorder":0}

        returnKwargs = getMessageProps("Done..","Nearest neighbors calculated and added to graph.")
        returnKwargs["lineCollections"] = lineCollections
        return returnKwargs

    def getSizeGroupsForScatter(self,dataID, sizeColumn = None, sizeColumnType = None, sizesSetByUser = None):
        ""
        #at the moment similiar to columnColor - merge
        if sizeColumn is None:
            sizeColumn = self.sizeColumn
        if sizeColumnType is None:
            sizeColumnType = self.sizeColumnType
        sizeColumnName = sizeColumn.values[0]
        sizeGroupData = pd.DataFrame(columns=["size","group"])

        categoryIndexMatch = None
        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,sizeColumn)["fnKwargs"]["data"]

        if sizeColumnType == "Integers":
            sizeColumnType, rawData = self._checkIntegerColumn(sizeColumn,rawData)

        if sizeColumnType == "Categories":
            if sizeColumn.index.size > 1:
                rawSizeData = rawData.apply(tuple,axis=1)
            else:
                rawSizeData = rawData[sizeColumnName]


            sizeCategories = rawSizeData.unique()
            nCategories = sizeCategories.size
            scaleSizes = np.linspace(0.2,1,num=nCategories,endpoint=True)
            sizeMap = dict(zip(sizeCategories,scaleSizes))
            sizeMap = replaceKeyInDict(self.sourceData.replaceObjectNan,sizeMap,0.1)

            scaledSizedata = rawSizeData.map(sizeMap)

            sizeData = scaledSizedata.values * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize

            sizeGroupData["size"] = scaleSizes * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize
            sizeGroupData["group"] = sizeCategories
            sizeGroupData["internalID"] = [getRandomString() for n in sizeGroupData.index]

            categoryIndexMatch = dict([(intID,rawData[rawSizeData == category].index) for category, intID in zip(sizeGroupData["group"].values,
                                                                                                                    sizeGroupData["internalID"].values)])
          #  print(sizeGroupData)
           # print(propsData)
        elif sizeColumnType == "Numeric Floats":
            transformFuncs = {"log2":np.log2,"ln":np.log,"log10":np.log10}
            if sizeColumn.size > 1:
                if self.aggMethod == "mean":
                    rawData = rawData.mean(axis=1)
                elif self.aggMethod == "sum":
                    rawData = rawData.sum(axis=1)
            fakeIndex = np.arange(rawData.index.size)
            nanIndex = fakeIndex[np.isnan(rawData.values).flatten()]
            sizeTransform = self.sourceData.parent.config.getParam("transform.numeric.size.columns")
            if sizeTransform in transformFuncs:
                rawValuesZeroNan = rawData.replace(0,np.nan)
                rawValues = transformFuncs[sizeTransform](rawValuesZeroNan.values)

            else:
                rawValues = rawData.values
            #get quanitles for normalization
            minV, q25, median, q75, maxV = np.nanquantile(rawValues, q = [0,0.25,0.5,0.75,1])
            #scale size
            sizeData = (rawValues.flatten() - minV) / (maxV - minV) * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize
            #set nan Size
            sizeData[nanIndex] = 0.1 * self.minScatterSize

            groupNames = ["Max ({})".format(getReadableNumber(maxV)),
                            "75% Quantile ({})".format(getReadableNumber(q75)),
                            "Median ({})".format(getReadableNumber(median)),
                            "25% Quantile ({})".format(getReadableNumber(q25)),
                            "Min ({})".format(getReadableNumber(minV)),
                            "NaN"]

            sizeGroupData["group"] = groupNames
            sizeGroupData["size"] = [round((x-minV)/(maxV-minV) * (self.maxScatterSize-self.minScatterSize) + self.minScatterSize,1) \
                                                                            for x in  [maxV,q75,median,q25,minV,0]] #0.1 == Nan Size

            sizeGroupData["internalID"] = [getRandomString() for n in sizeGroupData.index]

            sizeGroupData.loc[sizeGroupData["group"] == "NaN","size"] = round(0.1 * self.minScatterSize,1)


        propsData = pd.DataFrame(sizeData,columns=["size"], index=rawData.index)
        title = mergeListToString(sizeColumn.values,"\n")


        return {"sizeGroupData":sizeGroupData,
                "propsData":propsData,
                "title":title,
                "categoryIndexMatch":categoryIndexMatch,
                "categoryEncoded":"size",
                "encodedColumnNames" : sizeColumn,
                "isEditable":sizeColumnType == "Categories"}

    def getLinearRegression(self,dataID,numericColumnPairs):
        ""
        lineData = {}
        for n,numColumns in enumerate(numericColumnPairs):

            xList,yList,slope,intercept,rValue,pValue, stdErrorSlope  = self.sourceData.statCenter.runLinearRegression(dataID,list(numColumns))
            lineKwargs = {"xdata":xList,"ydata":yList}
            lineData[n] = lineKwargs

        funcProps = getMessageProps("Done..","Linear regression line added.")
        funcProps["lineData"] = lineData
        return funcProps

    def getLowessLine(self,dataID,numericColumnPairs):
        ""
        lineData = {}
        areaData = {}
        for n,numColumns in enumerate(numericColumnPairs):

            lowessFit = self.sourceData.statCenter.runLowess(dataID,list(numColumns))

            lineKwargs = {"xdata":lowessFit[:,0],"ydata":lowessFit[:,1]}
            lineData[n] = lineKwargs
            areaData[n] = {"x" : lowessFit[:,0], "y1": lowessFit[:,2], "y2":lowessFit[:,3], "facecolor":"lightgrey","edgecolor":"None","alpha":0.5}

        funcProps = getMessageProps("Done..","Lowess line added.")
        funcProps["lineData"] = lineData
        funcProps["areaData"] = areaData

        return funcProps

    def getColorGroupsForVolcanoModeScatter(self,dataID, significantColumns, colorColumns, numericColumns, columnPairs, significantStr = "+"):
        "Significant columns are exclusively categorical."

        colorForUpReg = self.sourceData.parent.config.getParam("scatter.volcano.color.sig.up")
        colorForDownReg = self.sourceData.parent.config.getParam("scatter.volcano.color.sig.down")
        colorForNonSig = self.sourceData.parent.config.getParam("scatter.volcano.color.non.sig")
        encodeNonSig = self.sourceData.parent.config.getParam("scatter.volcano.color.encode.non.sig")
        columnNames = pd.concat([significantColumns,numericColumns, colorColumns])

        sigUpString = f"{significantStr} & up"
        sigDownString = f"{significantStr} & down"


        rawData = self.sourceData.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]
        boolIdcs = rawData[significantColumns] == significantStr
        log2Columns = [columnPair[0] for columnPair in columnPairs]
        aboveZero = rawData.loc[:,log2Columns] > 0

        #crate significant categories
        fakeSignificantCategories = pd.DataFrame(np.full(shape=aboveZero.shape, fill_value="-", dtype=str), columns=boolIdcs.columns, index=boolIdcs.index)
        for n in range(len(log2Columns)):
            aboveZeroAndSignificant = np.logical_and(aboveZero.iloc[:,n].values,boolIdcs.iloc[:,n].values)
            belowAndSignificant = np.logical_and(~aboveZero.iloc[:,n].values,boolIdcs.iloc[:,n].values)
            fakeSignificantCategories.iloc[aboveZeroAndSignificant,n] = sigUpString
            fakeSignificantCategories.iloc[belowAndSignificant,n] = sigDownString

        if colorColumns.size > 0:
            if colorColumns.size > 1:
                colorCategories = rawData.loc[:,colorColumns].apply(tuple,axis=1).values #mege if more color columns given
            else:
                colorCategories = rawData.loc[:,colorColumns].values
            colorCategories = pd.DataFrame(np.tile(colorCategories.flatten().reshape(-1,1), (1, boolIdcs.columns.size)), columns=boolIdcs.columns, index=boolIdcs.index)


            potentialColorAndSignificantCategories = pd.DataFrame(
                {x: zip(fakeSignificantCategories.loc[:,x].values,colorCategories.loc[:,x].values) for x in fakeSignificantCategories.columns}, index = boolIdcs.index)
            uniqueCategories = np.unique(potentialColorAndSignificantCategories.values.flatten())
            categoricalGroups = dict( )
            it = itertools.groupby(uniqueCategories.tolist(), operator.itemgetter(0))

            for k, l in it:
                categoricalGroups[k] = list(l)

            # upCategories = [categoricalValue for categoricalValue in uniqueCategories if categoricalValue[0] == sigUpString]
            # downCategories = [categoricalValue for categoricalValue in uniqueCategories if categoricalValue[0] ==  downCategories]
            # nonSigCategoris = [categoricalValue for categoricalValue in uniqueCategories if categoricalValue[0] ==  downCategories]
            #customPalette = False, customPaletteKwargs
            colorsUp, layerMapUp = self.sourceData.colorManager.createColorMapDict(categoricalGroups[sigUpString],
                                                            addNaNLevels=[(self.sourceData.replaceObjectNan,)* (colorColumns.index.size+1)],
                                                            as_hex=True,
                                                            customPalette = True,
                                                            customPaletteKwargs = {"palette" : f"light:{colorForUpReg}_r","n_colors" : len(categoricalGroups[sigUpString])+1})
            colorsDown, layerMapDown = self.sourceData.colorManager.createColorMapDict(categoricalGroups[sigDownString],
                                                            addNaNLevels=[(self.sourceData.replaceObjectNan,)* (colorColumns.index.size+1)],
                                                            as_hex=True,
                                                            customPalette = True,
                                                            customPaletteKwargs = {"palette" : f"light:{colorForDownReg}_r","n_colors" : len(categoricalGroups[sigDownString])+1})
            if encodeNonSig:
                colorsNonSig, layerMapNonSig = self.sourceData.colorManager.createColorMapDict(categoricalGroups[self.sourceData.replaceObjectNan],
                                                            addNaNLevels=[(self.sourceData.replaceObjectNan,)* (colorColumns.index.size+1)],
                                                            as_hex=True,
                                                            customPalette = True,
                                                            customPaletteKwargs = {"palette" : f"light:{colorForNonSig}_r","n_colors" : len(categoricalGroups[self.sourceData.replaceObjectNan])})
            else:
                colorsNonSig = dict(zip(categoricalGroups[self.sourceData.replaceObjectNan],[self.sourceData.colorManager.nanColor]*len(categoricalGroups[self.sourceData.replaceObjectNan])))
                layerMapNonSig = {self.sourceData.colorManager.nanColor : -2}

            colors = {**colorsUp,**colorsDown, **colorsNonSig}
            layerMap = {**layerMapUp, **layerMapDown, **layerMapNonSig}
            #colors, layerMap = self.getColorMapDictAndLayers(uniqueCategories, colorColumns.index.size+1)
        else:
            potentialColorAndSignificantCategories = fakeSignificantCategories.copy()
            colors = {
                f"{significantStr} & up" : colorForUpReg,
                f"{significantStr} & down" : colorForDownReg,
                "-" : self.sourceData.colorManager.nanColor
            }
            layerMap = {
                colorForUpReg : 2,
                colorForDownReg : 2,
                self.sourceData.colorManager.nanColor : -1

            }

        colorData = potentialColorAndSignificantCategories.applymap(lambda x: colors[x])
        layerData = colorData.applymap(lambda x: layerMap[x])

        colorGroupData = pd.DataFrame(columns=["color","group"])
        colorGroupData["group"] = list(colors.keys())
        colorGroupData["color"] = list(colors.values())
        colorGroupData["internalID"] = [getRandomString() for _ in colorGroupData.index]


        scatterProps = {}
        categoryIndexMatch = {}
        textKwargs = {}
        for n,columnPair in enumerate(columnPairs):
            scatterProps[columnPair] = pd.DataFrame(columns=["color","layer"])
            # if n > 0 and boolIdcs.shape[1] == 1:
            #     colorDataForColumnPair =  colorData.iloc[:,0]
            #     layerValues = layerData.iloc[:,0]
            # else:
            #     colorDataForColumnPair =  colorData.iloc[:,n]
            #     layerValues = layerData.iloc[:,0]

            #colorDataForColumnPair[aboveZeroAndSignificant] = colorForUpReg
            scatterProps[columnPair]["color"] = colorData.iloc[:,n]
            scatterProps[columnPair]["layer"] = layerData.iloc[:,n]
            sumUp = np.sum(fakeSignificantCategories.iloc[:,n] == f"{significantStr} & up")
            sumDown = np.sum(fakeSignificantCategories.iloc[:,n] == f"{significantStr} & down")
            textKwargs[n] = [{ #here key must be n matching the axisDict keyword, not the scatterPlot columnpair
                    "x" : 0.98,
                    "y": 0.98,
                    "s" : f"{sumUp}",
                    "color" : colorForUpReg,
                    "ha" : "right",
                    "va" : "top"},
                {
                    "x" : 0.02,
                    "y": 0.98,
                    "s" : f"{sumDown}",
                    "color" : colorForDownReg,
                    "ha" : "left",
                    "va" : "top"
                    }]
            rawColorCategories = potentialColorAndSignificantCategories.iloc[:,n]


            categoryIndexMatch[columnPair] =  dict([(intID,rawColorCategories.index[[x == category for x in rawColorCategories.values]]) for category, intID in zip(colorGroupData["group"].values,
                                                                                                                 colorGroupData["internalID"].values)])

        tableTitle = mergeListToString(colorColumns.values,"\n") if colorColumns.size > 0 else "Significance"

        funcProps = getMessageProps("Done..","Color coding done ..")
        funcProps["propsData"] = scatterProps
        funcProps["title"] = tableTitle
        funcProps["colorGroupData"] = colorGroupData
        funcProps["categoryIndexMatch"] = categoryIndexMatch
        funcProps["texts"] = textKwargs
        funcProps["encodedColumnNames"] = colorColumns
       # print(funcProps)
        return funcProps

    def getColorGroupsDataForScatter(self,dataID, colorColumn = None, colorColumnType = None, colorGroupData = None, userMinMax = None):
        ""


        if colorColumn is None:
            if self.colorColumn is None:
                return {}
            colorColumn = self.colorColumn
        if colorColumnType is None:
            colorColumnType = self.colorColumnType
        colorColumnName = colorColumn.values[0]

        categoryIndexMatch = None

        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,colorColumn)["fnKwargs"]["data"]
        if colorColumnType == "Integers": #decide how to treat integers (e.g. as continous number or as a category.)
            uniqueValues = np.unique(rawData[colorColumn].values).size
            if uniqueValues > rawData.index.size * 0.75 / colorColumn.size:
                colorColumnType = "Numeric Floats"
                rawData = rawData.astype(float)
            else:
                colorColumnType = "Categories"
                rawData = rawData.astype(str)

        if colorColumnType == "Categories":

            if colorColumn.index.size > 1:
                rawColorData = rawData.apply(tuple,axis=1)
            else:
                rawColorData = rawData[colorColumnName]

            if colorGroupData is None:
                colorCategories = rawColorData.unique()
            else:
                colorCategories = colorGroupData["group"].values

            colors, layerMap = self.getColorMapDictAndLayers(colorCategories,colorColumn.index.size)
            colorData = pd.DataFrame(rawColorData.map(colors).values,
                                        columns=["color"],
                                        index=rawData.index)
            #add color layer props



            colorData["layer"] = colorData["color"].map(layerMap)

            if colorGroupData is None:
                colorGroupData = pd.DataFrame(columns=["color","group"])
                colorGroupData["group"] = [f"{colorCategory} ({np.sum(rawColorData == colorCategory)})" for colorCategory in colorCategories]
                colorGroupData["color"] = [colors[k] for k in colorCategories]
                #map color data and save as df
                colorGroupData["internalID"] = [getRandomString() for _ in colorGroupData.index]
               # categoryIndexMatch = dict([(intID,rawData[rawColorData.values == category].index) for category, intID in zip(colorGroupData["group"].values,
                #                                                                                                            colorGroupData["internalID"].values)])
            else:

                colorGroupData["color"] = [colors[k] for k in colorCategories]

            categoryIndexMatch = dict([(intID,rawData.index[rawColorData == category]) for category, intID in zip(colorCategories,
                                                                                                                colorGroupData["internalID"].values)])


        elif colorColumnType == "Numeric Floats":

            twoColorMap = self.sourceData.colorManager.colorMap
            if colorColumn.size > 1:
                if self.aggMethod == "mean":
                    rawData = rawData.mean(axis=1)
                elif self.aggMethod == "sum":
                    rawData = rawData.sum(axis=1)

            nanIndex = rawData.index[np.isnan(rawData.values).flatten()]

            minV, q25, median, q75, maxV = np.nanquantile(rawData.values,q = [0,0.25,0.5,0.75,1])
            if userMinMax is not None:
                minV, maxV = userMinMax
                median = (maxV+minV) / 2
                q25 = median - (median-minV)/2
                q75 = median + (maxV-median)/2


            colorValues, colorsInMapper = self.sourceData.colorManager.matchColorsToValues(arr = rawData.values.flatten(), colorMapName = twoColorMap, vmin = minV, vmax = maxV, asHex=True)
            colorData =  pd.DataFrame(colorValues,
                                    columns=["color"],
                                    index=rawData.index)
            colorData.loc[nanIndex,"color"] = self.sourceData.colorManager.nanColor

            #save colors for legend
            #scaledColorVs = [to_hex(cmap( (x - minV) / (maxV - minV))) for x in [maxV,q75,median,q25,minV]]
            legendColors, _ = self.sourceData.colorManager.matchColorsToValues(arr = [maxV,q75,median,q25,minV], colorMapName = twoColorMap, vmin = minV, vmax = maxV, asHex=True)
            colorLimitValues = legendColors.flatten().tolist() + [self.sourceData.colorManager.nanColor]

            colorGroupData = pd.DataFrame(columns=["color","group"])
            groupNames = [f"Max ({getReadableNumber(maxV)})",
                        f"75% Quantile ({getReadableNumber(q75)})",
                        f"Median ({getReadableNumber(median)})",
                        f"25% Quantile ({getReadableNumber(q25)})",
                        f"Min ({getReadableNumber(minV)})",
                        "NaN"]
            colorGroupData["color"] = colorLimitValues
            colorGroupData["group"] = groupNames

        tableTitle = mergeListToString(colorColumn.values,"\n") if userMinMax is None else "Custom Scale\n" + mergeListToString(colorColumn.values,"\n")
        #save data to enable fast update
        self.colorColumn = colorColumn
        self.colorColumnType = colorColumnType

        return {
            "colorGroupData":colorGroupData,
            "propsData":colorData,
            "title":tableTitle,
            "categoryIndexMatch":categoryIndexMatch,
            "categoryEncoded":"color",
            "encodedColumnNames" : colorColumn,
            "isEditable":colorColumnType == "Categories"}

    def getXYPlotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        "Returns plot properties for a XY plot"
        colorGroupsData = pd.DataFrame()
        axisLabels = {}
        axisLimits = {}
        linesByInternalID = {}
        #
        lines = {}
        lineKwargs = {}
        markerLines = {}
        markerKwargs = {}
        #
        markerLine = None
        markerProps = {}

        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        config =  self.sourceData.parent.config


        if config.getParam("xy.plot.stem.mode"):
            stemBottom = config.getParam("xy.plot.bottom.stem")

        if config.getParam("xy.plot.against.index") or len(numericColumns) == 1:
            idxColumnName = "ICIndex_{}".format(getRandomString())#ensures that there are no duplicates
            rawData[idxColumnName] = np.arange(rawData.index.size)
            numericColumnPairs = [(idxColumnName,columnName) for columnName in numericColumns]
        elif config.getParam("xy.plot.single.x"): #use only first numeric column
            numericColumnPairs = [(numericColumns[0],columnName) for columnName in numericColumns[1:]]
        else:
            numericColumnPairs = list(zip(numericColumns[0::2], numericColumns[1::2]))

        separatePairs = config.getParam("xy.plot.separate.column.pairs")
        axisPostions = getAxisPosition(1 if not separatePairs else len(numericColumnPairs))
        if len(categoricalColumns) == 0:
            colorValues = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(numericColumnPairs))
            colorGroupsData["group"] = ["{}:{}".format(*columnPair) if "ICIndex" not in columnPair[0] else "{}".format(columnPair[1]) for columnPair in numericColumnPairs]
            colorCategories = ["None"]
        else:
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns)
            numColorCategories = colorCategories.size
            colorGroupsData["group"] = colorCategories
            colorValues = self.sourceData.colorManager.getNColorsByCurrentColorMap(numColorCategories)
            colorGroupBy = rawData.groupby(by=categoricalColumns[0],sort=False)
        #line2D
        colorGroupsData["color"] = colorValues
        colorGroupsData["internalID"] = [getRandomString() for n in colorValues]
        colorGroupsData = colorGroupsData[["color","group","internalID"]]

        for n,columnPair in enumerate(numericColumnPairs):

            for m,colorCategory in enumerate(colorCategories):

                if len(categoricalColumns) == 0:
                    colorValue = colorValues[n]
                    data = rawData
                    internalID = colorGroupsData["internalID"].iloc[n]
                else:
                    data = colorGroupBy.get_group(colorCategory)
                    internalID = colorGroupsData["internalID"].iloc[m]
                    colorValue = colorValues[m]

                if config.getParam("xy.plot.stem.mode"):
                    #create LineCollections
                    lineSegments = [[(x,stemBottom),(x,y)] for x,y in data[list(columnPair)].values]
                    lineProps = dict(
                            linestyle = config.getParam("xy.plot.line.style"),
                            segments = lineSegments,
                            color = colorValue,
                            linewidth = config.getParam("xy.plot.linewidth"))

                    l = LineCollection(**lineProps)
                    if config.getParam("xy.plot.show.marker"):

                        markerProps = dict(
                            xdata = data[columnPair[0]].values,
                            ydata = data[columnPair[1]].values,
                            color = "None",
                            markeredgecolor = "darkgrey",
                            markerfacecolor = colorValue,
                            markeredgewidth = config.getParam("xy.plot.marker.edge.width"),
                            alpha = config.getParam("xy.plot.alpha"),
                            markersize = config.getParam("xy.plot.marker.size"),
                            linewidth = 0,
                            marker = config.getParam("xy.plot.marker"))

                        markerLine = Line2D(**markerProps)
                    #markerline, stemlines, baseline = stem(rawData[columnPair[0]].values,rawData[columnPair[1]].values, use_line_collection = True)

                else:

                    lineProps = dict(
                            xdata = data[columnPair[0]].values,
                            ydata = data[columnPair[1]].values,
                            color = colorValue,
                            markeredgecolor = "darkgrey",
                            markeredgewidth = config.getParam("xy.plot.marker.edge.width"),
                            alpha = config.getParam("xy.plot.alpha"),
                            markersize = config.getParam("xy.plot.marker.size"),
                            linewidth = config.getParam("xy.plot.linewidth"),
                            marker = config.getParam("xy.plot.marker") if config.getParam("xy.plot.show.marker") else "")

                    l = Line2D(**lineProps)


                if markerLine is not None:
                    if internalID in linesByInternalID:
                        linesByInternalID[internalID].extend([l,markerLine])
                    else:
                        linesByInternalID[internalID] = [l,markerLine]
                else:
                    if internalID in linesByInternalID:
                        linesByInternalID[internalID].append(l)
                    else:
                        linesByInternalID[internalID] = [l]

                if not separatePairs:
                    if 0 not in lines:
                        lines[0] = []
                        lineKwargs[0] = []
                        markerLines[0] = []
                        markerKwargs[0] = []
                    lines[0].append(l)
                    lineKwargs[0].append({"ID":internalID,"props":lineProps})
                    markerLines[0].append(markerLine)
                    markerKwargs[0].append({"ID":internalID,"props":markerProps})
                else:
                    if n not in lines:
                        lines[n] = [l]
                        lineKwargs[n] = [{"ID":internalID,"props":lineProps}]
                        markerKwargs[n] = [{"ID":internalID,"props":markerProps}]
                        markerLines[n] = [markerLine]
                    else:
                        lines[n].append(l)
                        lineKwargs[n].append({"ID":internalID,"props":lineProps})
                        markerKwargs[n].append({"ID":internalID,"props":markerProps})
                        markerLines[n].append(markerLine)



        if not separatePairs:
            xAxisColumns = np.unique([columnPair[0] for columnPair in numericColumnPairs])
            yAxisColumns = np.unique([columnPair[1] for columnPair in numericColumnPairs])
            xAxisMin, xAxisMax, yAxisMin, yAxisMax = self._getXYLimits(X = rawData[xAxisColumns].values.flatten(),
                                    Y  = rawData[yAxisColumns].values.flatten())

            axisLimits[0] = {"xLimit": (xAxisMin,xAxisMax), "yLimit": (yAxisMin,yAxisMax)}

            axisLabels[0] = {"x":numericColumnPairs[0][0],
                            "y":numericColumnPairs[0][1]} if len(numericColumnPairs) == 1 else {"x":"Index","y":"Y-value"}
        else:

            for n, columnPair in enumerate(numericColumnPairs):
                xColumn, yColumn = columnPair
                xAxisMin, xAxisMax, yAxisMin, yAxisMax = self._getXYLimits(
                                                    X = rawData[xColumn].values.flatten(),
                                                    Y  = rawData[yColumn].values.flatten())
                xAxisLabel = xColumn if not (config.getParam("xy.plot.against.index") or len(numericColumns) == 1) else "Index"
                axisLimits[n] = {"xLimit": (xAxisMin,xAxisMax), "yLimit": (yAxisMin,yAxisMax)}
                axisLabels[n] = {"x":xAxisLabel,"y":yColumn}

        return {"data":{
                    "lines":lines,
                    "lineKwargs" : lineKwargs,
                    "axisLabels" : axisLabels,
                    "axisPositions":axisPostions,
                    "dataColorGroups": colorGroupsData,
                    "colorCategoricalColumn": "Line Collection" if len(categoricalColumns) == 0 else categoricalColumns[0],
                    "axisLimits" : axisLimits,
                    "dataID" : dataID,
                    "linesByInternalID": linesByInternalID,
                    "markerKwargs": markerKwargs,
                    "markerLines": markerLines,
                    "numericColumnPairs" : numericColumnPairs,
                    "hoverData" : rawData
                    }}

    def _getXYLimits(self,X,Y,marginFrac = 0.1):
        ""
        minX, maxX = np.nanquantile(X,q = [0,1])
        marginX = np.sqrt(maxX**2 + minX**2)
        xAxisMin,  xAxisMax = minX - marginX * marginFrac, maxX + marginX * marginFrac

        minY, maxY = np.nanquantile(Y,q = [0,1])
        marginY = np.sqrt(maxY**2 + minY**2)
        yAxisMin,  yAxisMax = minY - marginY * marginFrac , maxY + marginY * marginFrac
        return xAxisMin, xAxisMax, yAxisMin, yAxisMax

    def getWordCloud(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""

        axisPostions = getAxisPosition(1)
        rawData = self.sourceData.getDataByColumnNames(dataID,categoricalColumns)["fnKwargs"]["data"]
        config =  self.sourceData.parent.config
        splitString = config.getParam("word.cloud.split_string")
        cmap = self.sourceData.colorManager.get_max_colors_from_pallete()

        wc = WordCloud(
            font_path = "Arial",
            max_font_size = config.getParam("word.cloud.max_font_size"),
            min_font_size = config.getParam("word.cloud.min_font_size"),
            normalize_plurals = config.getParam("word.cloud.normalize_plurals"),
            max_words = config.getParam("word.cloud.max_words"),
            colormap = cmap,
            include_numbers = config.getParam("word.cloud.include_numbers"),
            background_color = config.getParam("word.cloud.background_color"),
            width = 1600,
            height = 800)

        if config.getParam("word.cloud.categories_to_frequencies"):

            splitData = rawData[categoricalColumns[0]].astype("str").str.split(splitString, expand=True).values.flatten()
            countedValues = pd.Series(splitData).value_counts(normalize=True).to_dict()
            wordcloud = wc.generate_from_frequencies(countedValues)
        else:
            textInput = " ".join(rawData[categoricalColumns].values.flatten().astype(str))
            wordcloud = wc.generate(textInput)

        return {"data":{
                "cloud":wordcloud,
                "axisPositions":axisPostions}
                }
