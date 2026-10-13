from .helpers import *
from .helpers import _width_functions, _lv_box_ends, _lv_outliers

class CategoricalPlotMixin(object):
    """Bar, line, histogram, violin, boxen, box, point, swarm and cluster plots."""

    def getClusterLineplots(self,clusterLabels, data, numericColumns, columnName,*args,**kwargs):
        ""
        plotData = {}
        quickSelect = {}
        xValues = np.arange(len(numericColumns))
        groupby = clusterLabels.groupby(columnName,sort=False)
        for n, (clusterLabel, clusterData) in enumerate(groupby):
            plotData[n] = {}
            quickSelect[n] = {}
            plotData[n]["segments"] = [list(zip(xValues,data.loc[idx,:].values.flatten())) for idx in clusterData.index]
            quickSelect[n]["positions"] = np.arange(len(numericColumns))
            quickSelect[n]["x"] = [data.loc[clusterData.index,colName] for colName in data.columns]
        return plotData, quickSelect, groupby

    def getClusterBoxplots(self,clusterLabels, data, numericColumns, columnName,*args,**kwargs):
        ""
        plotData = {}
        quickSelect = {}
        for n, (clusterLabel, clusterData) in enumerate(clusterLabels.groupby(columnName,sort=False)):
            plotData[n] = {}
            plotData[n]["x"] = [data.loc[clusterData.index,colName] for colName in data.columns]
            plotData[n]["patch_artist"] = True
            plotData[n]["positions"] = np.arange(len(numericColumns))
        quickSelect = plotData.copy()
        return plotData, quickSelect

    def getClusterProps(self, dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""
        clusterCenters = None
        #get config
        config = self.sourceData.parent.config
        #get cluster method
        method = config.getParam("clusterplot.method")
        plottype = config.getParam("clusterplot.type")
        #color for distance

        # color group file
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])

        clusterLabels, data, model = self.sourceData.statCenter.runCluster(dataID,numericColumns,method,True)
        clusterLabels = clusterLabels.sort_values("Labels")
        columnName = "C({})".format(method)
        dataMinValue, dataMaxValue = np.nanquantile(data.values.flatten(),q=[0,1])
        clusterLabels = pd.DataFrame(["C({})".format(x) for x in clusterLabels.values.flatten()], index=clusterLabels.index,columns=[columnName])

        numNumericColumns = len(numericColumns)

        if method in ["kmeans","Birch"] and config.getParam("clusterplot.show.cluster.center"):

            if hasattr(model,"cluster_centers_"):
                clusterCenters = model.cluster_centers_
            elif hasattr(model,"subcluster_centers_"):
                clusterCenters = model.subcluster_centers_

            extraLines = dict([(n,{
                                "xdata":np.arange(numNumericColumns),
                                "ydata":clusterCenter.flatten(),
                                "linewidth":0.75,
                                "color":"black"}) for n,clusterCenter in enumerate(clusterCenters)])
        else:
            extraLines = {}
        #print(clusterLabels)
        uniqueClusters = clusterLabels[columnName].unique()
        nClusters = uniqueClusters.size
        displayDistanceByColor = config.getParam("clusterplot.lineplot.color.distance") and method in ["kmeans","Birch"] and plottype == "lineplot"
        #print(uniqueClusters)
        #print(nClusters)
        colorMap, _ = self.sourceData.colorManager.createColorMapDict(uniqueClusters,addNaNLevels=[-1],as_hex = True)
        #print(colorMap)
        if plottype == "boxplot":
            plotData,qSData = self.getClusterBoxplots(clusterLabels,data,numericColumns,columnName)
            faceColors = dict([(n,[colorMap[uniqueCluster]] * numNumericColumns) for n,uniqueCluster in enumerate(uniqueClusters)])

        elif plottype == "lineplot":

            plotData, qSData, groupby = self.getClusterLineplots(clusterLabels,data,numericColumns,columnName)
            if displayDistanceByColor:
                faceCs = []
                distanceMeasures = pd.DataFrame(model.transform(data.loc[clusterLabels.index]),index=clusterLabels.index)
               # cmap = self.sourceData.colorManager.get_max_colors_from_pallete(config.getParam("twoColorMap"))
                #cmap,colors = self.sourceData.colorManager.matchColorsToValues(distanceMeasures)
                distanceValues = np.concatenate([distanceMeasures.loc[clusterData.index,n].values.flatten() for n, (_, clusterData) in enumerate(groupby)])
                qs = np.quantile(distanceValues,q=[0,0.25,0.5,0.75,1])
                minValue,q25,median,q75, maxValue = qs
                groupNamesForLegend = self.getNumericColorMappingGroupNames(maxValue,q75,median,q25,minValue)
                colorArrayForLegend, _  = self.sourceData.colorManager.matchColorsToValues(np.flip(qs.flatten()),"Spectral",vmin=minValue,vmax=maxValue)
                colorArrayForLegendHex = [to_hex(c) for c in colorArrayForLegend]
                for n, (_, clusterData) in enumerate(groupby):
                    X = distanceMeasures.loc[clusterData.index,n].values
                    colorArray, _  = self.sourceData.colorManager.matchColorsToValues(X,"Spectral",vmin=minValue,vmax=maxValue)
                    faceCs.append((n,colorArray))

                faceColors = dict(faceCs)#dict([(n,colorMap[uniqueCluster]) ])

            else:

                faceColors = dict([(n,colorMap[uniqueCluster]) for n,uniqueCluster in enumerate(uniqueClusters)])

        axisPositions = getAxisPosition(nClusters,maxCol=self.maxColumns)
        axisLabels = dict([(n,{"x":"","y":"Value"}) for n in range(nClusters)])#
        axisTitles = dict([(n,"{} n:{}".format(uniqueCluster,np.sum(clusterLabels[columnName] == uniqueCluster))) for n,uniqueCluster in enumerate(uniqueClusters)])
        tickPositions = dict([(n,np.arange(len(numericColumns))) for n in range(nClusters)])#
        #tickLabels = dict([(n,[str(x) for x in np.arange(len(numericColumns))]) for n in range(nClusters)])#
        tickLabels = dict([(n,numericColumns) for n in range(nClusters)])#
        axisLimits = dict([(n,{"yLimit":[dataMinValue, dataMaxValue],"xLimit" : [-0.5,numNumericColumns-0.5]}) for n in range(nClusters)])#)



        if displayDistanceByColor:
            colorGroups["group"] = groupNamesForLegend
            colorGroups["color"] = colorArrayForLegendHex
            colorGroups["internalID"] = [getRandomString() for _ in range(len(groupNamesForLegend))]
        else:

            colorGroups["group"] = list(colorMap.keys())
            colorGroups["color"] = list(colorMap.values())
            colorGroups["internalID"] = [getRandomString() for _ in range(len(colorMap))]




        return {"data":{
            "plotType" : plottype,
            "plotData": plotData,
            "axisPositions" : axisPositions,
            "axisLabels"    :   axisLabels,
            "tickPositions": tickPositions,
            "tickLabels": tickLabels,
            "axisLimits" : axisLimits,
            "axisTitles" : axisTitles,
            "facecolors" : faceColors,
            "dataColorGroups" : colorGroups,
            "colorCategoricalColumn" : "Cluster Labels" if not displayDistanceByColor else "Cluster Distance",
            "clusterLabels" : clusterLabels,
            "dataID" : dataID,
            "extraLines" : extraLines,
            "quickSelect" : qSData
        }}

    def getBarplotProps(self, dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""
        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")
        plotData, \
        axisPositions, \
        boxPositions, \
        tickPositions, \
        tickLabels, \
        colorGroups, \
        faceColors, \
        colorCategoricalColumn, \
        xWidth, axisLabels, axisLimits, axisTitles, groupNames, verticalLines, groupedPlotData  = calculatePositions(dataID,self.sourceData,numericColumns,categoricalColumns,self.maxColumns,splitByCategories= splitByCats)


        filteredData = OrderedDict()
        hoverData = OrderedDict()
        for n,plotData in plotData.items():
            data = plotData["x"]
            hoverData[n] = {"x" : data}
            plotData["height"] = [np.mean(x) for x in data]
            plotData["width"] = xWidth * (self.sourceData.parent.config.getParam("barplot.width.scale"))
            plotData["color"] = faceColors[n]
            if "CI" in self.barplotError:
                ci = extractCIFromstring(self.barplotError)
                yerrorValues = [CI(a.values,ci) for a in data]
            elif self.barplotError == "Std":
                yerrorValues = [np.std(a.values) for a in data]
            plotData["yerr"] = None if all(not np.isfinite(x) for x in yerrorValues) else yerrorValues
            plotData["error_kw"] = {"capsize":rcParams["errorbar.capsize"],"elinewidth":0.5,"markeredgewidth":0.5,"zorder":1e6}

            plotData["x"] = boxPositions[n]
            filteredData[n] = plotData

        return {"data":{
                "plotData":filteredData,#"
                "facecolors" : faceColors,
                "axisPositions":axisPositions,
                "hoverData" : hoverData,
                "tickLabels": tickLabels,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisLimits" : axisLimits,
                "axisTitles" : axisTitles,
                "groupNames" : groupNames,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "verticalLines" : verticalLines,
                "colorCategoricalColumn" : colorCategoricalColumn,
                #"tooltipsTexts" : texts,
                "dataID":dataID}}

    def getLineplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        minQ = np.inf
        maxQ = -np.inf
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        axisTitles = {}

        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        if len(categoricalColumns) == 0:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
            quantiles = np.nanquantile(data.values,[0,0.25,0.5,0.75,1],axis=0)
            xValues = np.arange(len(numericColumns))
            plotData = {0:[{"quantiles":quantiles,"xValues":xValues,"color":self.sourceData.colorManager.nanColor}]}
            minQ, maxQ = np.min(quantiles[0,:]),np.max(quantiles[-1,:])
            marginY = np.sqrt(minQ**2 + maxQ**2) * 0.05
            nNumCol = len(numericColumns)
            marginX = 0.05*(nNumCol-1)
            axisLimits = {0:{"xLimit":[-marginX,len(numericColumns)-(1-marginX)],"yLimit":[minQ-marginY,maxQ+marginY]}}
            colorCategoricalColumn = ""
            colorGroups["color"] = [self.sourceData.colorManager.nanColor]
            colorGroups["group"] = [""]
            colorGroups["internalID"] = [getRandomString()]
            hoverData = {0:data}
            tickPositions = {0:np.arange(nNumCol)}
            tickLabels = {0:numericColumns}


        elif len(categoricalColumns) == 1:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
            plotData = {0:[]}
            colorCategoricalColumn = categoricalColumns[0]
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = colorCategoricalColumn)
           # colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(groupBy.keys()))
            colors,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            for groupName, groupData in data.groupby(by=categoricalColumns[0],sort=False):
                quantiles = np.nanquantile(groupData[numericColumns],[0,0.25,0.5,0.75,1],axis=0)
                xValues = np.arange(len(numericColumns))
                plotData[0].append({"quantiles":quantiles,"xValues":xValues,"color":colors[groupName]})
                minQGroup, maxQGroup = np.nanmin(quantiles[0,:]),np.nanmax(quantiles[-1,:])
                if minQGroup < minQ:
                    minQ = minQGroup
                if maxQGroup > maxQ:
                    maxQ = maxQGroup
            nNumCol = len(numericColumns)
            marginY = np.sqrt(minQ**2 + maxQ**2) * 0.05
            marginX = 0.05*(nNumCol-1)
            axisLimits = {0:{"xLimit":[-marginX,len(numericColumns)-(1-marginX)],"yLimit":[minQ-marginY,maxQ+marginY]}}
            tickPositions = {0:np.arange(nNumCol)}
            tickLabels = {0:numericColumns}

            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(colorCategories.size)]

            hoverData = {0:data}

        elif len(categoricalColumns) == 2:
            subplotBorders = dict(wspace=0.10, hspace = 0.1,bottom=0.05,right=0.95,top=0.95)
            splitCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            axisPositions = getAxisPosition(splitCategories.size,maxCol=self.maxColumns)
            plotData = dict([(n,[]) for n in axisPositions.keys()])

            colorCategoricalColumn = categoricalColumns[0]
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = colorCategoricalColumn)
           # colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(groupBy.keys()))
            colors,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            hoverData = {}

            for n, (axisName,axisData) in enumerate(data.groupby(by=categoricalColumns[1],sort=False)):
                #n = axis number
                hoverData[n] = axisData
                for groupName, groupData in axisData.groupby(by=categoricalColumns[0],sort=False):
                    quantiles = np.nanquantile(groupData[numericColumns],[0,0.25,0.5,0.75,1],axis=0)
                    xValues = np.arange(len(numericColumns))
                    plotData[n].append({"quantiles":quantiles,"xValues":xValues,"color":colors[groupName]})
                    minQGroup, maxQGroup = np.nanmin(quantiles[0,:]),np.nanmax(quantiles[-1,:])
                    if minQGroup < minQ:
                        minQ = minQGroup
                    if maxQGroup > maxQ:
                        maxQ = maxQGroup

                nNumCol = len(numericColumns)
                marginY = np.sqrt(minQ**2 + maxQ**2) * 0.05
                marginX = 0.05*(nNumCol-1)

                axisLimits = dict([(n,{"xLimit":[-marginX,len(numericColumns)-(1-marginX)],
                                       "yLimit":[minQ-marginY,maxQ+marginY]}) for n in axisPositions.keys()])
                tickPositions = dict([(n,np.arange(nNumCol)) for n in axisPositions.keys()])
                tickLabels = dict([(n, numericColumns) for n in axisPositions.keys()])
                axisTitles[n] = "{}: {}".format(categoricalColumns[1],axisName)
                colorGroups["color"] = colors.values()
                colorGroups["group"] = colorCategories
                colorGroups["internalID"] = [getRandomString() for n in range(colorCategories.size)]






        return {"data":{
                "plotData":plotData,
                "hoverData" : hoverData,
                "axisPositions":axisPositions,
                "axisTitles" : axisTitles,
                "subplotBorders":subplotBorders,
                "dataColorGroups": colorGroups,
                "axisLimits":axisLimits,
                "tickPositions" : tickPositions,
                "tickLabels" : tickLabels,
                "numericColumns":numericColumns,
                "colorCategoricalColumn" : colorCategoricalColumn}
                }

    def getHistogramProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        #get raw data
        patches = {}
        axisLimits = {}
        axisLabels = {}
        hoverData = {}

        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        subplotBorders = dict(wspace=0.175, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        axisPositions = getAxisPosition(len(numericColumns),maxCol=self.maxColumns)
        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]

        if len(categoricalColumns) == 0:

            colors = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(numericColumns))
            internalIDs = [getRandomString() for n in range(len(colors))]
            for n,numCol in enumerate(numericColumns):
                if self.histogramLog:
                    histData = np.log2(data[numCol]).replace([np.inf, -np.inf],np.nan).dropna()
                else:
                    histData = data[numCol].replace([np.inf, -np.inf], np.nan).dropna()
                if histData.empty:
                        continue
                hist, bins = np.histogram(histData.values,
                                            bins=self.histogramBins,
                                            density=self.histogramDensity)
                width = 0.95 * (bins[1] - bins[0])

               # center = (bins[:-1] + bins[1:]) / 2
                if self.histogramCumulative:
                    hist = np.cumsum(hist)
                axisLabels[n] = {"x":numCol if not self.histogramLog else "log2({})".format(numCol)}
                hoverData[n] = histData
                if self.histogramHisttype == "bar":
                    patches[n] = [{
                            "p":dict(
                                xy = (binStart,0),
                                width= width,
                                height= hist[m],
                                facecolor = colors[n],
                                linewidth = 0.2,
                                edgecolor = "black"),
                            "type": "Rectangle",
                            "internalID" : internalIDs[n]
                                                    } for m,binStart in enumerate(bins[:-1])]
                elif self.histogramHisttype == "step":

                    xy = self.createSteppedHistoValues(bins,hist)
                    patches[n] = [{"p":dict(
                                        xy = xy,
                                        edgecolor = colors[n],
                                        linewidth = self.histogramLinewidth,
                                        closed= False,
                                        facecolor = "None"
                                        ),
                                    "type":"Polygon",
                                    "internalID" : internalIDs[n]}]

                colorGroups["color"] = colors
                colorGroups["group"] = numericColumns
                colorGroups["internalID"] = internalIDs
                colorCategoricalColumn = "Numeric Columns"
                minHist, maxHist = np.min(hist), np.max(hist)
                xMargin = np.sqrt(bins[-1]**2 + bins[0]**2) * 0.02
                yMargin = np.sqrt(minHist**2 + maxHist**2) * 0.02
                axisLimits[n] = {
                    "xLimit":(bins[0]-xMargin ,bins[-1]+xMargin),
                    "yLimit":(0,maxHist + yMargin)}

        elif len(categoricalColumns) > 0:
            colorCategoricalColumn = "\n".join(categoricalColumns)
            groupby = data.groupby(by=categoricalColumns, sort=self.histogramSortCategories)
            colorCategories = list(groupby.groups.keys()) #self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = colorCategoricalColumn)
            internalIDs = OrderedDict([(colorCat,getRandomString()) for colorCat in colorCategories])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories,
                                                            as_hex=True,
                                                            addNaNLevels=[(self.sourceData.replaceObjectNan,)*len(categoricalColumns)],)

            for n,numCol in enumerate(numericColumns):
                minHist, maxHist = np.inf, -np.inf
                if self.histogramLog:
                    histRangeMin, histRangeMax = np.nanmin(np.log2(data[numCol].values)), np.nanmax(np.log2(data[numCol].values))
                else:
                    histRangeMin, histRangeMax = np.nanmin(data[numCol].values), np.nanmax(data[numCol].values)

                for groupName, groupData in groupby:

                    internalID = internalIDs[groupName]
                    if self.histogramLog:
                        histData = np.log2(groupData[numCol]).replace([np.inf, -np.inf],np.nan).dropna()
                    else:
                        histData = groupData[numCol].dropna()
                    hoverData[n] = histData
                    if histData.empty:
                        continue
                    hist, bins = np.histogram(
                                histData.values,
                                bins=self.histogramBins,
                                density=self.histogramDensity,
                                range = (histRangeMin, histRangeMax))
                    width = 0.95 * (bins[1] - bins[0])
                # center = (bins[:-1] + bins[1:]) / 2
                    if self.histogramCumulative:
                        hist = np.cumsum(hist)
                    axisLabels[n] = {"x": numCol if not self.histogramLog else "log2({})".format(numCol)}
                    if self.histogramHisttype == "bar":

                        histPatches = [{
                                "p":dict(
                                    xy = (binStart,0),
                                    width= width,
                                    height= hist[m],
                                    facecolor = colors[groupName],
                                    linewidth = 0.2,
                                    edgecolor = "black",
                                    alpha = .5),
                                "type" : "Rectangle",
                                "internalID" : internalID
                                                        } for m,binStart in enumerate(bins[:-1])]

                    elif self.histogramHisttype == "step":

                        xy = self.createSteppedHistoValues(bins,hist)
                        histPatches = [{"p":dict(
                                            xy = xy,
                                            edgecolor = colors[groupName],
                                            linewidth = self.histogramLinewidth,
                                            closed= False,
                                            facecolor = "None"
                                            ),
                                        "type" : "Polygon",
                                        "internalID" : internalIDs[groupName]}]
                    if n in patches:
                        p = patches[n]
                        patches[n] = p + histPatches
                    else:
                        patches[n] = histPatches

                    minHistCol, maxHistCol = np.nanmin(hist), np.nanmax(hist)

                    if minHistCol < minHist:
                        minHist = minHistCol
                    if maxHistCol > maxHist:
                        maxHist = maxHistCol

                    xMargin = np.sqrt(bins[-1]**2 + bins[0]**2) * 0.02
                    yMargin = np.sqrt(minHist**2 + maxHist**2) * 0.02

                    axisLimits[n] = {
                        "xLimit":(bins[0]-xMargin ,bins[-1]+xMargin),
                        "yLimit":(0,maxHist + yMargin)}


            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = internalIDs.values()



        return {"data":{
            "patches" : patches,
            "axisLabels" : axisLabels,
            "axisPositions" : axisPositions,
            "axisLimits" : axisLimits,
            "subplotBorders" : subplotBorders,
            "dataColorGroups": colorGroups,
            "colorCategoricalColumn" : colorCategoricalColumn,
            "hoverData" : hoverData
        }}

    def createSteppedHistoValues(self, bins, hist):
        ""
        x = np.repeat(bins[1:],2)
        x = np.append([bins[0]],x).reshape(-1,1)
        y = np.repeat(hist,2)
        y = np.append(y,[hist[-1]]).reshape(-1,1)

        return np.append(x,y,axis=1)

    def getViolinProps(self,dataID,numericColumns, categoricalColumns,*args,**kwargs):
        ""
        # axisPostions = dict([(n,[1,1,n+1]) for n in range(1)])
        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)

        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")
        #data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        #colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
        plotData, \
        axisPositions, \
        violinPositions, \
        tickPositions, \
        tickLabels, \
        colorGroups, \
        faceColors, \
        colorCategoricalColumn, \
        xWidth, axisLabels, axisLimits, axisTitles, groupNames, verticalLines, groupedPlotData  = calculatePositions(
                                                                                dataID,
                                                                                self.sourceData,
                                                                                numericColumns,
                                                                                categoricalColumns,
                                                                                self.maxColumns,
                                                                                splitByCategories= splitByCats)
        # xLabel = ""

        filteredData = OrderedDict()
        medianData = OrderedDict()
        minMaxLine = OrderedDict()
        quantileLine = OrderedDict()
        hoverData = OrderedDict()

        for n,plotData in plotData.items():
            data = plotData["x"]
            plotData["widths"] = xWidth
            plotData["dataset"] = [x.values for x in data]

            plotData["showmeans"] = self.sourceData.parent.config.getParam("violin.show.means")
            plotData["positions"] = violinPositions[n]
            plotData["showextrema"] = self.sourceData.parent.config.getParam("violin.show.extrema")
            plotData["points"] = self.sourceData.parent.config.getParam("violin.points")
            #plotData["quantiles"] = [0,0.1,0.2,0.5,0.7,0.9]
            hoverData[n] = {"x" : data }
            del plotData["x"]
            filteredData[n] = plotData

            medianData[n] = {}
            medianData[n]["x"] = violinPositions[n]
            medianData[n]["y"] = [np.median(yData) for yData in plotData["dataset"]]

            minMaxLine[n] = [np.append(np.array([x]),np.nanquantile(yData,q = [0,1])) for x,yData in zip(violinPositions[n],plotData["dataset"])]
            quantileLine[n] = [np.append(np.array([x]),np.nanquantile(yData,q = [0.25,0.75])) for x,yData in zip(violinPositions[n],plotData["dataset"])]

        return {"data":{
                "plotData":filteredData,#"
                "facecolors" : faceColors,
                "medianData" : medianData,
                "minMaxLine" :  minMaxLine,
                "quantileLine" : quantileLine,
                "axisPositions":axisPositions,
                "tickLabels": tickLabels,
                "axisTitles" : axisTitles,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisLimits" : axisLimits,
                "groupNames" : groupNames,
                "hoverData" : hoverData,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "verticalLines" : verticalLines,
                "colorCategoricalColumn" : colorCategoricalColumn,
                "dataID":dataID}}

    def getBoxenProps(self, dataID, numericColumns, categoricalColumns,*args,**kwargs):
        """
        Part of this code was adapted from the great seaborn package!
        """
        def height(b):
                return b[1] - b[0]

        def vertRectProps(x, b, i, k, w,widths):
            rect = dict(
                    xy = (x - widths * w / 2, b[0]),
                    width = widths * w,
                    height = height(b),
                    fill=True
                    )
            return rect

        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        showMeans = self.sourceData.parent.config.getParam("boxen.show.means")
        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")

        plotCalcData, axisPositions, boxPositions, tickPositions, tickLabels, colorGroups, \
            faceColors, colorCategoricalColumn, xWidth, axisLabels, axisLimits, \
                axisTitles, groupNames, verticalLines, groupedPlotData  = calculatePositions(dataID,
                                                                    self.sourceData,
                                                                    numericColumns,
                                                                    categoricalColumns,
                                                                    self.maxColumns,
                                                                    splitByCategories = splitByCats)

        rectData = OrderedDict()
        hoverData = OrderedDict()
        meanData = OrderedDict()
        positionData = OrderedDict()
        outlierData = OrderedDict()
        widthFn = self.sourceData.parent.config.getParam("boxen.width.calculation")
        xWidth = xWidth * self.sourceData.parent.config.getParam("boxenplot.width.scale")
        for n,plotData in plotCalcData.items():
            data = plotData["x"]
            hoverData[n] = {"x" : data}
            positionData[n] = []
            rectData[n] = []
            outlierData[n] = []
            meanData[n] = {"x":[],"y":[]}
            for m,boxData in enumerate(data):
                outlierProps = {}
                hexColor = faceColors[n][m]
                internalID = colorGroups.loc[colorGroups["color"] == hexColor,"internalID"].values[0]
                x = boxPositions[n][m]
                positionData[n].append(x)
                if boxData.size == 0:
                    rectData[n].append({
                        "rectProps":[],
                        "cmap":None,
                        "medianLine":{},
                        "internalID":internalID,
                        "outlierProps" : outlierProps})

                elif boxData.size == 1:
                    y = boxData.iloc[0]
                    medianLineProps = {
                        "xdata" : [x-xWidth/2,x+xWidth/2],
                        "ydata" : [y,y],
                        "linewidth" : 0.5,
                        "color" : "black",
                        "solid_capstyle" : "butt"
                    }
                    rectData[n].append({
                            "rectProps":[],
                            "cmap":None,
                            "medianLine":{},
                            "internalID":internalID,
                            "outlierProps" : outlierProps})
                    if showMeans:
                        meanData[n]["x"].append(x)
                        meanData[n]["y"].append(y)

                else:

                    #fncs form seaborn package
                    vals = boxData.values.flatten()
                    boxEnds, k = _lv_box_ends(vals)
                    width = _width_functions(widthFn)

                    # Scale the width of the boxes so the biggest starts at 1
                    w_area = np.array([width(height(b), i, k)
                                    for i, b in enumerate(boxEnds)])

                    w_area = w_area / np.max(w_area)


                    outliers = _lv_outliers(vals,k)
                    if len(outliers) > 0:
                        xdata = np.full(len(outliers), x)
                        ydata = outliers

                        outlierProps.update({
                            "xdata" : xdata,
                            "ydata" : ydata,
                            "linewidth" : 0,
                            "markerfacecolor" : "black",
                            "markeredgecolor" : "black",
                            "markeredgewidth" : 0.5,
                            "markersize" : 3,
                            "marker" : "+"
                         })

                    boxes = [vertRectProps(x, b[0], i, k, b[1],xWidth)
                        for i, b in enumerate(zip(boxEnds, w_area))]

                    y = np.median(boxData)

                    medianLineProps = {
                        "xdata" : [x-xWidth/2,x+xWidth/2],
                        "ydata" : [y,y],
                        "linewidth" : 0.5,
                        "color" : "black",
                        "solid_capstyle" : "butt"
                    }

                    if showMeans:
                        meanData[n]["x"].append(x)
                        meanData[n]["y"].append(np.nanmean(vals))
                       # meanData[n].append({"x":[x],"":[np.nanmean(vals)]})

                    # Construct a color map from the input color
                    rgb = [hexColor, (1, 1, 1)]
                    cmap = LinearSegmentedColormap.from_list('new_map', rgb)
                    # Make sure that the last boxes contain hue and are not pure white
                    rgb = [hexColor, cmap(.85)]
                    cmap = LinearSegmentedColormap.from_list('new_map', rgb)

                    rectData[n].append({"rectProps":boxes,
                            "cmap":cmap,
                            "medianLine":medianLineProps,
                            "internalID":internalID,
                            "faceColor" : hexColor,
                            "outlierProps" : outlierProps
                            })

                outlierData[n].append(outlierProps)


        return {"data":{
                "plotData":rectData,#"
                "meanData" : meanData,
                "positions" : positionData,
                "hoverData" : hoverData,
                "facecolors" : faceColors,
                "axisPositions":axisPositions,
                "axisLimits" : axisLimits,
                "tickLabels": tickLabels,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisTitles" : axisTitles,
                "groupNames" : groupNames,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "colorCategoricalColumn" : colorCategoricalColumn,
                "verticalLines" : verticalLines,
                "groupedPlotData" : groupedPlotData,
                "dataID":dataID}}

    def getBoxplotProps(self, dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""

        subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        if len(categoricalColumns) > 3:
            splitByCats = False
        else:
            splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")

        plotCalcData, axisPositions, boxPositions, tickPositions, tickLabels, colorGroups, \
            faceColors, colorCategoricalColumn, xWidth, axisLabels, axisLimits, \
                axisTitles, groupNames, verticalLines, groupedPlotData  = calculatePositions(dataID,
                                                                    self.sourceData,
                                                                    numericColumns,
                                                                    categoricalColumns,
                                                                    self.maxColumns,
                                                                    splitByCategories = splitByCats)

        filteredData = OrderedDict()
        for n,plotData in plotCalcData.items():
            plotData["widths"] = xWidth * self.sourceData.parent.config.getParam("boxplot.width.scale")
            plotData["patch_artist"] = True
            plotData["positions"] = boxPositions[n]
            plotData["capprops"] = {"linewidth":rcParams["boxplot.whiskerprops.linewidth"]}
            filteredData[n] = plotData


        return {"data":{
                "plotData":filteredData,#"
                "facecolors" : faceColors,
                "axisPositions":axisPositions,
                "axisLimits" : axisLimits,
                "tickLabels": tickLabels,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisTitles" : axisTitles,
                "groupNames" : groupNames,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "colorCategoricalColumn" : colorCategoricalColumn,
                "verticalLines" : verticalLines,
                "groupedPlotData" : groupedPlotData,
                #"tooltipsTexts" : texts,
                "dataID":dataID}}

    def getPointplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        """
        If categoricalColmns == 0



        return general  line2Ds
        """

        scaleXAxis = self.sourceData.parent.config.getParam("scale.numeric.x.axis")
        splitString = self.sourceData.parent.config.getParam("split.string.x.category")
        splitIndex = self.sourceData.parent.config.getParam("split.string.index")
        faceAndLineColorSame = self.sourceData.parent.config.getParam("pointplot.line.marker.same.color")
        lineWidth = self.sourceData.parent.config.getParam("pointplot.line.width")
        errorColorAsLineColor = self.sourceData.parent.config.getParam("pointplot.error.bar.color.as.line")
        errorEdgeColorAsLineColor = self.sourceData.parent.config.getParam("pointplot.edgecolor.as.line")
        errorLineWidth = self.sourceData.parent.config.getParam("pointplot.error.line.width")
        axisLimits = {}
        tickLabels = {}
        tickPositions = {}
        axisLabels = {}
        hoverData = {}
        #
        lineKwargs = OrderedDict()
        errorKwargs = OrderedDict()
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        markerSize = self.sourceData.parent.config.getParam("pointplot.marker.size")

        #get raw data
        rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        if len(categoricalColumns) == 0:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
            colorList = self.sourceData.colorManager.getNColorsByCurrentColorMap(N=len(numericColumns))

            hoverData[0] = rawData

            for n in axisPositions.keys():
                columnMeans = rawData[numericColumns].mean().values
                ci = extractCIFromstring(self.barplotError)
                errorValues = [CI(rawData[columnName].dropna(),ci) for columnName in numericColumns]
                errorValues = [e if not np.isnan(e) else 0 for e in errorValues]
                maxErrorValue = np.nanmax(errorValues)
                minValue, maxValue = np.nanmin(columnMeans), np.nanmax(columnMeans)

                if np.isnan(maxErrorValue):
                    maxErrorValue = 0.05*maxValue

                tickLabels[n] = numericColumns
                tickPositions[n] = np.arange(len(numericColumns))

                line2DKwargs = []
                line2DErrorKwargs = []
                for m in np.arange(len(numericColumns)):
                    line2D = {}
                    error2D = {}
                    line2D["marker"] = "o"
                    line2D["color"] = colorList[m] if faceAndLineColorSame else "black"
                    line2D["linewidth"] = lineWidth
                    line2D["markerfacecolor"] = colorList[m]
                    line2D["markeredgecolor"] = colorList[m] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = [m]
                    error2D["y"] = [columnMeans[m]]
                    error2D["yerr"] = [errorValues[m]]
                    error2D["elinewidth"] = errorLineWidth

                    error2D["ecolor"] = colorList[m] if errorColorAsLineColor else rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                #define axis limits

                #marginY = np.sqrt(((maxValue+3*maxErrorValue)-(minValue-2*maxErrorValue))**2)

                axisLimits[n] = {
                        "xLimit": (-0.5,len(numericColumns)-0.5),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }
                colorGroups["color"] = colorList
                colorGroups["group"] = numericColumns
                colorGroups["internalID"] = [getRandomString() for n in range(len(numericColumns))]
                axisLabels[n] = {"x":"","y":"Value"}
            colorCategoricalColumn = "Numeric Columns"

        elif len(categoricalColumns) == 1 and len(numericColumns) == 1:

            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)
            #get unique categories
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]
            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns)
            colorCategoricalColumn = categoricalColumns[0]
            hoverData[0] = rawData
            ci = extractCIFromstring(self.barplotError)
            for n in axisPositions.keys():
                columnMeans = [data.loc[:,numericColumns[0]].mean() for groupName, data in groupByCatColumn]

                errorValues = [CI(data[numericColumns[0]].dropna(),ci) for groupName, data in groupByCatColumn]
                maxErrorValue = np.nanmax(errorValues)
                minValue, maxValue = np.nanmin(columnMeans), np.nanmax(columnMeans)
                if np.isnan(maxErrorValue):
                    maxErrorValue = 0.05*maxValue
                tickLabels[n] = colorCategories
                if scaleXAxis:
                    try:
                        tickPos = [float(x.split(splitString)[splitIndex]) for x in colorCategories]
                    except:
                        tickPos = np.arange(nColorCats)
                else:
                    tickPos = np.arange(nColorCats)

                tickPositions[n] = tickPos

                line2DKwargs = []
                line2DErrorKwargs = []
                for m, category in enumerate(colorCategories):
                    line2D = {}
                    error2D = {}
                    line2D["marker"] = "o"
                    line2D["color"] = colors[category] if faceAndLineColorSame else "black"
                    line2D["linewidth"] = lineWidth
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = colors[category] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = [tickPos[m]]
                    error2D["y"] = [columnMeans[m]]
                    error2D["yerr"] = [errorValues[m]]
                    error2D["elinewidth"] = errorLineWidth

                    error2D["ecolor"] = colors[category] if errorColorAsLineColor else rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                #define axis limits
                minX, maxX = np.min(tickPos), np.max(tickPos)
                distance = np.sqrt(minX**2 + maxX**2) * 0.05
                axisLimits[n] = {
                        "xLimit": (minX-distance,maxX+distance),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }
                axisLabels[n] = {"x":"colorCategoricalColumn","y":"value"}

        elif len(categoricalColumns) == 1 and len(numericColumns) > 1:
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(1,maxCol=self.maxColumns)

            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]

            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)

            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)
            hoverData[0] = rawData

            if np.isnan(maxErrorValue):
                maxErrorValue = 0.05*maxValue
            colorCategoricalColumn = categoricalColumns[0]

            for n in axisPositions.keys():
                tickLabels[n] = numericColumns

                tickPositions[n] = np.arange(len(numericColumns))
                line2DKwargs = []
                line2DErrorKwargs = []
                for m,category in enumerate(colorCategories):

                    line2D = {}
                    error2D = {}
                    line2D["marker"] = "o"
                    line2D["color"] = colors[category] if faceAndLineColorSame else "black"
                    line2D["linestyle"] = "-"
                    line2D["linewidth"] = lineWidth
                    line2D["aa"] = True
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = colors[category] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = np.arange(len(numericColumns))#line2D["xdata"]
                    error2D["y"] = np.array([meanErrorData[category][numColumn]["value"] for numColumn in numericColumns])#groupMeans.loc[category ,:].values#line2D["ydata"]
                    error2D["yerr"] = np.array([meanErrorData[category][numColumn]["error"] for numColumn in numericColumns])
                    error2D["elinewidth"] = errorLineWidth

                    error2D["ecolor"] = colors[category] if errorColorAsLineColor else rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                axisLimits[n] = {
                        "xLimit": (-0.5,len(numericColumns)-0.5),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }
                axisLabels[n] = {"x":"","y":"value"}

        elif len(categoricalColumns) == 2 and len(numericColumns) >= 1:

            nNumCols = len(numericColumns)
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(nNumCols,maxCol=self.maxColumns)

            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in range(nColorCats)]
            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)
            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)
            if np.isnan(maxErrorValue):
                maxErrorValue = 0.05*maxValue



            for n in axisPositions.keys():
                tickLabels[n] = tickCats
                if scaleXAxis:
                    try:
                        tickPos = [float(x.split(splitString)[splitIndex]) for x in tickCats]
                    except:
                        tickPos = np.arange(tickCats.size)
                else:
                    tickPos = np.arange(tickCats.size)
                tickPositions[n] = tickPos
                xValues = dict([(catName,tickPos[nCat]) for nCat,catName in enumerate(tickCats)])
                numColumn = numericColumns[n]

                minX, maxX = np.min(tickPos), np.max(tickPos)
                distance = np.sqrt(minX**2 + maxX**2) * 0.05
                axisLimits[n] = {
                        "xLimit": (minX-distance,maxX+distance),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }

                line2DKwargs = []
                line2DErrorKwargs = []

                for m,category in enumerate(colorCategories):

                    x = []
                    y = []
                    e = []

                    for tCat in tickCats:

                        groupName = (tCat,category)
                        if groupName in meanErrorData:
                            if tCat in xValues:
                                x.append(xValues[tCat])
                            else:
                                x.append(np.nan)
                            y.append(meanErrorData[(tCat,category)][numColumn]["value"])# groupMeans.loc[(tCat,category),numColumn])
                            e.append(meanErrorData[(tCat,category)][numColumn]["error"])#groupErrors.loc[(tCat,category),numColumn])
                        else:
                            continue
                    line2D = {}
                    error2D = {}

                    line2D["marker"] = "o"
                    line2D["color"] = colors[category] if faceAndLineColorSame else "black"
                    line2D["linestyle"] = "-"
                    line2D["linewidth"] = lineWidth
                    line2D["markerfacecolor"] = colors[category]
                    line2D["markeredgecolor"] = colors[category] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                    line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                    line2D["markersize"] = markerSize
                    line2DKwargs.append(line2D)

                    error2D["x"] = x
                    error2D["y"] = y
                    error2D["yerr"] = e
                    error2D["elinewidth"] = errorLineWidth

                    error2D["ecolor"] = colors[category] if errorColorAsLineColor else rcParams["patch.edgecolor"]
                    line2DErrorKwargs.append(error2D)
                lineKwargs[n] = line2DKwargs
                errorKwargs[n] = line2DErrorKwargs
                axisLabels[n] = {"x":categoricalColumns[0],"y":numColumn}
                hoverData[n] = rawData[numColumn]
            colorCategoricalColumn = categoricalColumns[1]



        elif len(categoricalColumns) == 3 and len(numericColumns) >= 1:
            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            #catgeory dividend into different axes
            axisCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[2])
            #first category splis data on x axis
            xAxisCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            #second category is color coded
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #number of numeric columns
            NNumCol = len(numericColumns)
            #create axis
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            axisPositions = getAxisPosition(n = axisCategories.size *  NNumCol, maxCol = axisCategories.size)
            nColorCats = colorCategories.size
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            #get tick categories
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            colorGroups["color"] = colors.values()
            colorGroups["group"] = colorCategories
            colorGroups["internalID"] = [getRandomString() for n in colors.values()]
            #some calculataions for axis limits
            globalMin, globalMax = np.nanquantile(rawData[numericColumns].values, q = [0,1])
            #create groupby

            axisGroupby = rawData.groupby(categoricalColumns[2],sort=False)

            groupByCatColumn = self.sourceData.getGroupsbByColumnList(dataID,categoricalColumns,as_index=False)
            meanErrorData, minValue, maxValue, maxErrorValue = self.getCIForGroupby(groupByCatColumn,numericColumns)

            if np.isnan(maxErrorValue):
                maxErrorValue = 0.05*maxValue

            xValues = dict([(catName,x) for x,catName in enumerate(tickCats)])
            nAxis = -1
            for numColumn in numericColumns:

                for axisCat, axisCatData in axisGroupby:
                    line2DKwargs = []
                    line2DErrorKwargs = []
                    nAxis += 1
                    tickLabels[nAxis] = tickCats
                    if scaleXAxis:
                        try:
                            tickPos = [float(x.split(splitString)[splitIndex]) for x in tickCats]
                        except:
                            tickPos = np.arange(tickCats.size)
                    else:
                        tickPos = np.arange(tickCats.size)

                    tickPositions[nAxis] = tickPos
                    minX, maxX = np.min(tickPos), np.max(tickPos)
                    distance = np.sqrt(minX**2 + maxX**2) * 0.05
                    hoverData[nAxis] = axisCatData

                    axisLimits[nAxis] = {
                        "xLimit": (minX-distance,maxX+distance),
                        "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
                        }

                    for m,colorCategory in enumerate(colorCategories):

                            x = []
                            y = []
                            e = []
                            for xAxisCat in tickCats:
                                group = (colorCategory,xAxisCat,axisCat)
                                if group in meanErrorData:
                                    if xAxisCat in xValues:
                                        x.append(xValues[xAxisCat])
                                    else:
                                        x.append(np.nan)
                                    y.append(meanErrorData[group][numColumn]["value"])
                                    e.append(meanErrorData[group][numColumn]["error"])
                                else:
                                    continue
                            line2D = {}
                            error2D = {}

                            line2D["marker"] = "o"
                            line2D["color"] =  colors[colorCategory] if faceAndLineColorSame else "black"
                            line2D["linestyle"] = "-"
                            line2D["linewidth"] = lineWidth
                            line2D["markerfacecolor"] = colors[colorCategory]
                            line2D["markeredgecolor"] = colors[colorCategory] if errorEdgeColorAsLineColor else rcParams["patch.edgecolor"]
                            line2D["markeredgewidth"] = rcParams["patch.linewidth"]
                            line2D["markersize"] = np.sqrt(self.sourceData.parent.config.getParam("scatterSize"))
                            line2DKwargs.append(line2D)

                            error2D["x"] = x
                            error2D["y"] = y
                            error2D["yerr"] = e
                            error2D["elinewidth"] = errorLineWidth

                            error2D["ecolor"] = colors[colorCategory] if errorColorAsLineColor else rcParams["patch.edgecolor"]
                            line2DErrorKwargs.append(error2D)
                    lineKwargs[nAxis] = line2DKwargs
                    errorKwargs[nAxis] = line2DErrorKwargs
                    axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
            colorCategoricalColumn = categoricalColumns[0]
    # for n in axisPositions.keys():
    #                 tickLabels[n] = tickCats
    #                 tickPositions[n] = np.arange(tickCats.size)
    #                 numColumn = numericColumns[n]
    #                 axisLimits[n] = {
    #                         "xLimit": (-0.5,tickCats.size-0.5),
    #                         "yLimit" : (minValue-1.5*maxErrorValue,maxValue+1.5*maxErrorValue)
    #                         }

    #                 line2DKwargs = []
    #                 line2DErrorKwargs = []




        return {"data":{
                "plotData":lineKwargs,#"
                "errorData":errorKwargs,
                "axisPositions":axisPositions,
                "tickLabels": tickLabels,
                "tickPositions": tickPositions,
                "axisLabels" : axisLabels,
                "axisLimits" : axisLimits,
                "dataColorGroups": colorGroups,
                "subplotBorders":subplotBorders,
                "colorCategoricalColumn" : colorCategoricalColumn,
                "hoverData" : hoverData,
                #"tooltipsTexts" : texts,
                "dataID":dataID}}

    def getCIForGroupby(self,groupby, numericColumns):
        ""
        meanErrorData = OrderedDict()
        es = []
        maxValue = np.nanmax(groupby.max()[numericColumns].values)
        minValue = np.nanmin(groupby.min()[numericColumns].values)
        ci = extractCIFromstring(self.barplotError)
        for groupName, data in groupby:
            meanErrorData[groupName] = {}
            for numColumn in numericColumns:
                X = data[numColumn].dropna()
                if not X.empty:
                    errorValue = CI(X.values,ci )
                    m = np.nanmean(X.values)
                    if np.isnan(errorValue): #no confidence interval can be caluclated.
                        errorValue = 0

                    es.append(errorValue)
                    meanErrorData[groupName][numColumn] = {"value":m,
                                                           "error":errorValue}
                else:
                    meanErrorData[groupName][numColumn] = {"value":np.nan,
                                                           "error":np.nan}
        maxError = np.nanmax(errorValue)
        return meanErrorData, minValue, maxValue, maxError

    def _getBoxplotTooltips(self,data,numericColumns):
        ""
        tooltipsStr = {}
        desc = data[numericColumns].describe()
        for columnName in numericColumns:
            baseStr = ""
            for idx in desc.index:
                if baseStr != "":
                    baseStr += "\n"
                baseStr += "{} : {}".format(idx,getReadableNumber(desc.loc[idx,columnName]))
            tooltipsStr[columnName] = baseStr
        return tooltipsStr

    def getProteinPeptideProps(self, proteinDataID, peptideDataID, proteinColumnNames, peptideColumnNames):
        ""

    def getBoxplotProps2(self, dataID, numericColumns, categoricalColumns):
        ""
        boxplotProps = []
        xAxisLabels = []
        outliers = np.array([]).reshape(-1,2)

        if categoricalColumns is None or len(categoricalColumns) == 0:
            colorMap  = self.getColorMapper(numericColumns)
            data = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
            quantiles, IQR, upperBound, lowerBound = self._getBoxProps(data.values)

            x0, width = self.findPositions(numericColumns,categoricalColumns,self.plotType)

            for n,numericColumn in enumerate(numericColumns):

                outlierBool = (data[numericColumn] > upperBound[n]) | (data[numericColumn] < lowerBound[n])
                limits = [data.loc[outlierBool == False,numericColumn].min(), data.loc[outlierBool == False,numericColumn].max()]
                xCenter = x0[n] + width/2
                boxplotProp = {"IQR":IQR[n],"median":quantiles[1,n],"limits":limits,"y0": quantiles[0,n],"x0":x0[n],"width":width,"color":colorMap[numericColumn]}
                boxplotProps.append(boxplotProp)

                nOutliers = np.sum(outlierBool)
                outlierPositions = np.full(shape=(nOutliers,2),fill_value=xCenter)
                outlierPositions[:,1] = data.loc[outlierBool,numericColumn].values

                outliers = np.append(outliers,outlierPositions,axis=0)
                xAxisLabels.append((xCenter,numericColumn))



        return {"newPlot":True,"data":{"plotData":boxplotProps,"outliers":outliers,"xAxisLabels":[xAxisLabels],"plotType":"boxplot"}}

    def _getBoxProps(self,arr):
        ""
        quantiles = np.nanquantile(arr,q=[0.25, 0.5, 0.75], axis=0)
        IQR = quantiles[2,:] - quantiles[0,:]
        upperBound = quantiles[2,:] + 1.5 * IQR
        lowerBound = quantiles[0,:] - 1.5 * IQR

        return quantiles, IQR, upperBound, lowerBound

    def getSwarmplotProps(self,dataID,numericColumns,categoricalColumns,*args,**kwargs):
        ""
        numCatColumns = len(categoricalColumns)
        numNumColumns = len(numericColumns)
        #subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        multiScatterKwargs = {}
        colorCategoryIndexMatch = {}
        interalIDColumnPairs = {}
        axisTitles = {}
        axisLimits = {}
        tickPositions = {}
        numericColumnPairs = {}
        axisLabels = {}
        tickLabels = {}

        NProcesses = self.sourceData.parent.config.getParam("n.processes.multiprocessing")
        splitByCats = self.sourceData.parent.config.getParam("boxplot.split.data.on.category")
        replaceObjectNan = self.sourceData.replaceObjectNan
        if numCatColumns > 3: splitByCats = False
        colorGroupsData = pd.DataFrame()
        sizeGroupsData = pd.DataFrame()



        if numCatColumns > 0 and not splitByCats:
            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            axisPostions = getAxisPosition(n = numNumColumns, maxCol=2)

            uniqueValueIndex = {}
            tickPositionByUniqueValue = {}
            tickPositions = {}
            boxPositions = {}
            groupNames = {}
            plotData = pd.DataFrame(rawData, index = rawData.index)

            uniqueValuesByCatColumns = OrderedDict([(categoricalColumn,[cat for cat in rawData[categoricalColumn].unique() if cat != replaceObjectNan]) for categoricalColumn in categoricalColumns])
            uniqueValuesForCatColumns = [uniqueValuesByCatColumns[categoricalColumn]  for categoricalColumn in categoricalColumns]
            uniqueValuesPerCatColumn = dict([(categoricalColumn,uniqueValuesForCatColumns[n]) for n,categoricalColumn in enumerate(categoricalColumns)])
            uniqueCategories = ["Complete"] + ["{}:({})".format(uniqueValue,k) for k,v in uniqueValuesByCatColumns.items() for uniqueValue in v]
            colors,_ = self.sourceData.colorManager.createColorMapDict(uniqueCategories, as_hex=True)
            flatUniqueValues = ["Complete"] + [uniqueValue for sublist in uniqueValuesForCatColumns for uniqueValue in sublist]
            #drop "-"
            totalNumUniqueValues = np.array(uniqueValuesForCatColumns).flatten().size

            widthBox = 1/(totalNumUniqueValues + 1)
            border = widthBox/10

            colorGroupsData["color"] = colors.values()
            colorGroupsData["group"] = uniqueCategories
            colorGroupsData["internalID"] = [getRandomString() for n in colorGroupsData["color"].values]

            colorCategoricalColumn = "\n".join(categoricalColumns)

            offset = 0 + border + widthBox/2
            tickPositionByUniqueValue["Complete"] = offset

            for categoricalColumn, uniqueValues in uniqueValuesPerCatColumn.items():
                uniqueValueIndex[categoricalColumn] = {}
                offset += widthBox/2 #extra offset by categorical column

                for uniqueValue in uniqueValues:
                    offset += widthBox
                    idxBool = rawData[categoricalColumn] == uniqueValue
                    uniqueValueIndex[categoricalColumn][uniqueValue] = idxBool
                    tickPositionByUniqueValue["{}:({})".format(uniqueValue,categoricalColumn)] = offset
                    # if not uniqueValue == uniqueValues[-1]:
                    #     offset += widthBox/5
            offset += border + widthBox/2
            for n,numericColumn in enumerate(numericColumns):
            #init lists to stare props
                columnNames = []
                filteredData = []
                boxPositions[n] = []
                tickPositions[n] = []
               # faceColors[n] = []
                groupNames[n] = []
                interalIDColumnPairs[n] = dict()
                multiScatterKwargs[n] = dict()
                # add complete data
                X = rawData[numericColumn].dropna()
                filteredData.append(X)
                boxPositions[n].append(tickPositionByUniqueValue["Complete"] )
                tickPositions[n].append(tickPositionByUniqueValue["Complete"] )
                xName = f"{numericColumn}:Complete"
                #faceColors[n].append(colors["Complete"])
                groupNames[n].append("{}:Complete".format(numericColumn))
                #add complete data
                kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(rawData[[numericColumn]].dropna(),bandwidth = 0.75)
                columnPair = (xName,numericColumn)
                columnNames.extend([xName,numericColumn])
                multiScatterKwargs[n][columnPair] = {"color": colors["Complete"]}
                kdeData = distKDEData(kdeData) #distribute kde data
                kdeData = scaleBetween(kdeData,(0,widthBox/2))
                kdeData = kdeData + tickPositionByUniqueValue["Complete"] - widthBox/4
                data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                internalID = colorGroupsData.loc[colorGroupsData["group"] == "Complete"]["internalID"].values[0]
                if internalID not in colorCategoryIndexMatch:
                    colorCategoryIndexMatch[internalID] = kdeIndex
                else:
                    #because of nan removal, we have to join indices
                    colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")
                if internalID not in interalIDColumnPairs[n]:
                    interalIDColumnPairs[n][internalID] = [columnPair]
                else:
                    interalIDColumnPairs[n][internalID].append(columnPair)
                #add data to plot.
                plotData = plotData.join(data)
                #iterate through unique values
                for categoricalColumn, uniqueValues in uniqueValuesPerCatColumn.items():

                    for m,uniqueValue in enumerate(uniqueValues):
                        xName = f"x({numericColumn}:{categoricalColumn}:{uniqueValue})"
                        colorKey ="{}:({})".format(uniqueValue,categoricalColumn)
                        fc = colors[colorKey]
                        idxBool = uniqueValueIndex[categoricalColumn][uniqueValue]
                        uniqueValueFilteredData = rawData[numericColumn].loc[idxBool].dropna()
                        tickBoxPos = tickPositionByUniqueValue[colorKey]


                        if uniqueValueFilteredData.index.size > 0:

                            kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(pd.DataFrame(uniqueValueFilteredData),bandwidth = 0.75)
                            kdeData = scaleBetween(kdeData,(0,widthBox/2))
                            kdeData = distKDEData(kdeData) #distribute kde data
                            kdeData = kdeData + tickBoxPos
                            data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                            filteredData.append(uniqueValueFilteredData)
                            boxPositions[n].append(tickBoxPos)
                            tickPositions[n].append(tickBoxPos)
                            groupNames[n].append("{}-{}".format(numericColumn,colorKey))

                            plotData = plotData.join(data)
                            columnNames.extend([xName,numericColumn])
                            columnPair = (xName,numericColumn)
                            multiScatterKwargs[n][columnPair] = {"color": fc}
                            #saving internal id and color matches with pandas index (to allow easy manipulation)

                            internalID = colorGroupsData.loc[colorGroupsData["group"] == colorKey]["internalID"].values[0]
                            if internalID not in colorCategoryIndexMatch:
                                colorCategoryIndexMatch[internalID] = kdeIndex
                            else:
                                #because of nan removal, we have to join indices
                                colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")

                            if internalID not in interalIDColumnPairs[n]:
                                interalIDColumnPairs[n][internalID] = [columnPair]
                            else:
                                interalIDColumnPairs[n][internalID].append(columnPair)

                numericColumnPairs[n] = tuple(columnNames)
                #plotData[n] = {"x":filteredData}
            nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
            axisPostions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
            tickLabels = dict([(n,flatUniqueValues) for n in range(numNumColumns)])
            axisLabels = dict([(n,{"x":"Categories","y":numericColumn}) for n,numericColumn in enumerate(numericColumns)])
            axisLimits = dict([(n,{"xLimit" : (0,offset),"yLimit":None}) for n in range(numNumColumns)])


        elif numCatColumns == 0:
            #get raw data
            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns)["fnKwargs"]["data"]
            numericColumnPairs = []
            widthBox = 0.75
            tickPositions = {0:np.arange(numNumColumns) + widthBox}
            positions = np.arange(numNumColumns)
            colorDict,_ = self.sourceData.colorManager.createColorMapDict(numericColumns, as_hex=True)
            plotData = pd.DataFrame(rawData, index = rawData.index)
            columnNames = []
            multiScatterKwargs[0] = dict() #0 = axis id
            interalIDColumnPairs[0] = dict() #0 = axis id
            colorGroupsData["color"] = list(colorDict.values())
            colorGroupsData["group"] = numericColumns
            colorGroupsData["internalID"] = [getRandomString() for _ in range(numNumColumns)]

            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * numNumColumns
            sizeGroupsData["group"] = numericColumns
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values

            if rawData.size > 100000 and rawData.shape[1] > 4:
                #if big data, use multiprocessing, otherwise not
                with Pool(NProcesses) as p:
                    r = p.starmap(kdeCalc,[(rawData[[numColumn]].dropna(),0.75,"gaussian",widthBox,tickPositions[0][n],numColumn,n)  for n,numColumn in enumerate(numericColumns)])
                    kdeIndex, dataList, xNames = zip(*r)
                # print(kdeIndex, dataList, xNames)
                plotData = plotData.join(dataList)
                for n,(xName,numColumn) in enumerate(xNames):
                    columnNames.extend([xName,numColumn])
                    multiScatterKwargs[0][(xName,numColumn)] = {"color": colorDict[numColumn]}
                    internalID = colorGroupsData.loc[colorGroupsData["group"] == numColumn]["internalID"].values[0]
                    colorCategoryIndexMatch[internalID] = kdeIndex[n]
                    if internalID not in interalIDColumnPairs[0]:
                        interalIDColumnPairs[0][internalID] = [(xName,numColumn)]
                    else:
                        interalIDColumnPairs[0][internalID].append((xName,numColumn))

            else:
                for n,numColumn in enumerate(numericColumns):
                    xName = "x({})".format(numColumn)
                    groupData = rawData[[numColumn]].dropna()

                    if groupData.empty:
                        continue
                    elif groupData.index.size == 1:
                        kdeData = np.array([0]) +  positions[n]
                        data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                        kdeIndex = data.index
                    else:

                        #get kernel data
                        kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                        #get random x position around 0 to spread data

                        allSame = np.all(kdeData == kdeData[0])
                        if allSame:
                            kdeData = np.zeros(shape=kdeData.size)
                        else:
                            kdeData = scaleBetween(kdeData,(0,widthBox/2))

                        kdeData = distKDEData(kdeData)
                        #print(time.time()-t1,"numpy")
                        kdeData = kdeData + positions[n]
                        #save data

                        data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])


                    plotData = plotData.join(data)
                    columnNames.extend([xName,numColumn])
                    multiScatterKwargs[0][(xName,numColumn)] = {"color": colorDict[numColumn]}
                    internalID = colorGroupsData.loc[colorGroupsData["group"] == numColumn]["internalID"].values[0]
                    colorCategoryIndexMatch[internalID] = kdeIndex
                    if internalID not in interalIDColumnPairs[0]:
                        interalIDColumnPairs[0][internalID] = [(xName,numColumn)]
                    else:
                        interalIDColumnPairs[0][internalID].append((xName,numColumn))

            numericColumnPairs = {0:tuple(columnNames)}
            nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
            axisPostions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
            axisLabels = dict([(0,{"x":"Numeric Column(s)","y":"Value"})])
            tickLabels = dict([(0,numericColumns)])
            tickPositions = dict([(0,np.arange(len(numericColumns)))])
            colorCategoricalColumn = "Numeric Columns"

        elif numCatColumns == 1:

            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            numericColumnPairs = []

            plotData = pd.DataFrame(rawData, index = rawData.index)
            columnNames = []

            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            nColorCats = colorCategories.size
            colorDict,_ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            colorCategoricalColumn = categoricalColumns[0]
            catGroupby = rawData.groupby(categoricalColumns[0], sort=False)
            multiScatterKwargs[0] = dict()
            interalIDColumnPairs[0] = dict() #0 = axis ID

            globalMin, globalMax = np.nanquantile(rawData[numericColumns].values, q = [0,1])

            yMargin = np.sqrt(globalMax**2 + globalMin**2)*0.05
            colorGroupsData["color"] = colorDict.values()
            colorGroupsData["group"] = colorCategories
            colorGroupsData["internalID"] = [getRandomString() for _ in range(colorCategories.size)]

            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * nColorCats
            sizeGroupsData["group"] = colorCategories
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values

            tickPositions = []
            tickLabels = []
            widthBox= 1/(nColorCats)
            border = widthBox/3

            for m,numColumn in enumerate(numericColumns):

                startPos = m if m == 0 else m + (border * m) #add border
                endPos = startPos + widthBox * (nColorCats-1)
                positions = np.linspace(startPos,endPos,num=nColorCats)

                tickPos = np.median(positions)
                tickPositions.append(tickPos)
                tickLabels.append(numColumn)

                for nColCat, colCat in enumerate(colorCategories):
                    #remove nan for kernel estimation
                    groupData = catGroupby.get_group(colCat).dropna(subset=[numColumn])
                    if groupData.index.size > 0:
                        xName = "x({}:{})".format(numColumn,colCat)
                        #get kernel data
                        if groupData.index.size < 5:
                            kdeData = np.array([0]*groupData.index.size) +  positions[nColCat]
                            data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                            kdeIndex = data.index
                        else:
                            kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                            #get random x position around 0 to spread data between - and + kdeData
                            kdeData = scaleBetween(kdeData,(0,widthBox/2))

                            kdeData = distKDEData(kdeData)
                            kdeData = kdeData + positions[nColCat]
                            data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                        plotData = plotData.join(data)
                        columnNames.extend([xName,numColumn])
                        multiScatterKwargs[0][(xName,numColumn)] = {"color": colorDict[colCat]}
                        internalID = colorGroupsData.loc[colorGroupsData["group"] == colCat]["internalID"].values[0]
                        #saving internal id and color matches with pandas index (to allow easy manipulation)
                        if internalID not in colorCategoryIndexMatch:
                            colorCategoryIndexMatch[internalID] = kdeIndex
                        else:
                            #because of nan removal, we have to join indices
                            colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")

                        if internalID not in interalIDColumnPairs[0]:
                            interalIDColumnPairs[0][internalID] = [(xName,numColumn)]
                        else:
                            interalIDColumnPairs[0][internalID].append((xName,numColumn))


            numericColumnPairs = {0:tuple(columnNames)}

            nrows,ncols,subplotBorders = self._findScatterSubplotProps(numericColumnPairs)
            axisPostions = dict([(n,[nrows,ncols,n+1]) for n in range(len(numericColumnPairs))])
            axisLabels = dict([(n,{"x":"Numeric Column(s)","y":"Value"}) for n in range(1)])
            tickLabels = {0:tickLabels}
            axisLimits[0] = {"xLimit":[-2*border,endPos+2*border],"yLimit":[globalMin-yMargin,globalMax+yMargin]}
            tickPositions = {0:tickPositions}
            #print(tickPositions,tickLabels)

        elif numCatColumns == 2:

            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            globalMin, globalMax = np.nanquantile(rawData[numericColumns].values, q = [0,1])
            yMargin = np.sqrt(globalMax**2 + globalMin**2)*0.05

            #get color cats
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #matching colors to categories (dict)
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            nXAxisCats = tickCats.size
            colorCategoricalColumn = categoricalColumns[0]
            #save colorGroups data
            colorGroupsData["color"] = colors.values()
            colorGroupsData["group"] = colorCategories
            colorGroupsData["internalID"] = [getRandomString() for n in range(nColorCats)]

            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * nColorCats
            sizeGroupsData["group"] = colorCategories
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values

            axisPostions = getAxisPosition(len(numericColumns),maxCol=self.maxColumns)
            catGroupby = rawData.groupby(categoricalColumns, sort=False)

            #plot data
            plotData = pd.DataFrame(rawData, index = rawData.index)
            widthBox= 1/(nColorCats)
            border = widthBox / 3

            for nAxis,numColumn in enumerate(numericColumns):
                multiScatterKwargs[nAxis] = dict()
                interalIDColumnPairs[nAxis] = dict() #nAxis = axis ID
                catTickPositions = []
                catTickLabels = []
                columnNames = []

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
                            xName = "x({}:{}:{})".format(numColumn,colCat,tickCat)
                            if groupData.index.size == 1:
                                kdeData = np.array([0]) +  positions[nColCat]
                                data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                                kdeIndex = data.index
                            else:
                                #get kernel data

                                kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                                #get random x position around 0 to spread data between - and + kdeData
                                allSame = np.all(kdeData == kdeData[0])
                                if allSame:
                                    kdeData = np.zeros(shape=kdeData.size)
                                else:
                                    kdeData = scaleBetween(kdeData,(0,widthBox/2))

                                kdeData = distKDEData(kdeData)
                                kdeData = kdeData + positions[nColCat]
                                data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                            plotData = plotData.join(data)
                            columnNames.extend([xName,numColumn])
                            multiScatterKwargs[nAxis][(xName,numColumn)] = {"color": colors[colCat]}
                            #saving internal id and color matches with pandas index (to allow easy manipulation)
                            internalID = colorGroupsData.loc[colorGroupsData["group"] == colCat]["internalID"].values[0]
                            if internalID not in colorCategoryIndexMatch:
                                colorCategoryIndexMatch[internalID] = kdeIndex
                            else:
                                #because of nan removal, we have to join indices
                                colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")

                            if internalID not in interalIDColumnPairs[nAxis]:
                                interalIDColumnPairs[nAxis][internalID] = [(xName,numColumn)]
                            else:
                                interalIDColumnPairs[nAxis][internalID].append((xName,numColumn))

                tickPositions[nAxis] = catTickPositions
                numericColumnPairs[nAxis] = tuple(columnNames)
                axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
                tickLabels[nAxis] = catTickLabels
                axisLimits[nAxis] = {"yLimit":[globalMin-yMargin,globalMax+yMargin],"xLimit":[0-widthBox/2-border,nXAxisCats+widthBox/2+border]}
                #saving internal id and color matches with pandas index (to allow easy manipulation)

        elif numCatColumns == 3:

            rawData = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
            subplotBorders = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
            globalMin, globalMax = np.nanquantile(rawData[numericColumns].values, q = [0,1])
            yMargin = np.sqrt(globalMax**2 + globalMin**2)*0.05

            #get color cats
            colorCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[0])
            #matching colors to categories (dict)
            colors, _ = self.sourceData.colorManager.createColorMapDict(colorCategories, as_hex=True)
            nColorCats = colorCategories.size
            tickCats = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[1])
            axisCategories = self.sourceData.getUniqueValues(dataID = dataID, categoricalColumn = categoricalColumns[2])
            nXAxisCats = tickCats.size
            colorCategoricalColumn = categoricalColumns[0]
            #save colorGroups data
            colorGroupsData["color"] = colors.values()
            colorGroupsData["group"] = colorCategories
            colorGroupsData["internalID"] = [getRandomString() for n in range(nColorCats)]

            sizeGroupsData["size"] = [self.sourceData.parent.config.getParam("scatterSize")] * nColorCats
            sizeGroupsData["group"] = colorCategories
            sizeGroupsData["internalID"] = colorGroupsData["internalID"].values

            axisPostions = getAxisPosition(n = axisCategories.size *  numNumColumns, maxCol = axisCategories.size)
            catGroupby = rawData.groupby(categoricalColumns, sort=False)

            #plot data
            plotData = pd.DataFrame(rawData, index = rawData.index)
            widthBox= 1/(nColorCats)
            border = widthBox / 3


            tickPositions = {}
            numericColumnPairs = {}
            axisLabels = {}
            tickLabels = {}

            nAxis = -1

            for n,numColumn in enumerate(numericColumns):

                for nAxisCat, axisCat in enumerate(axisCategories):
                    nAxis +=1
                    multiScatterKwargs[nAxis] = dict()
                    interalIDColumnPairs[nAxis] = dict() #nAxis = axis ID
                    catTickPositions = []
                    catTickLabels = []
                    columnNames = []

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
                            if groupData.index.size > 0:
                                xName = "x({}:{}:{}:{})".format(numColumn,colCat,tickCat,axisCat)
                                #get kernel data
                                if groupData.index.size == 1:
                                    kdeData = np.array([0])+  positions[nColCat]
                                    data = pd.DataFrame(kdeData ,index=groupData.index, columns = [xName])
                                    kdeIndex = data.index
                                else:
                                    kdeData, kdeIndex = self.sourceData.getKernelDensityFromDf(groupData[[numColumn]],bandwidth = 0.75)
                                    #get random x position around 0 to spread data between - and + kdeData
                                    allSame = np.all(kdeData == kdeData[0])
                                    if allSame:
                                        kdeData = np.zeros(shape=kdeData.size)
                                    else:
                                        kdeData = scaleBetween(kdeData,(0,widthBox/2))

                                    kdeData = distKDEData(kdeData)
                                    kdeData = kdeData + positions[nColCat]
                                    data = pd.DataFrame(kdeData, index = kdeIndex, columns=[xName])
                                plotData = plotData.join(data)
                                columnNames.extend([xName,numColumn])
                                multiScatterKwargs[nAxis][(xName,numColumn)] = {"color": colors[colCat]}
                                #saving internal id and color matches with pandas index (to allow easy manipulation)
                                internalID = colorGroupsData.loc[colorGroupsData["group"] == colCat]["internalID"].values[0]
                                if internalID not in colorCategoryIndexMatch:
                                    colorCategoryIndexMatch[internalID] = kdeIndex
                                else:
                                    #because of nan removal, we have to join indices
                                    colorCategoryIndexMatch[internalID] = colorCategoryIndexMatch[internalID].join(kdeIndex, how="outer")

                                if internalID not in interalIDColumnPairs[nAxis]:
                                    interalIDColumnPairs[nAxis][internalID] = [(xName,numColumn)]
                                else:
                                    interalIDColumnPairs[nAxis][internalID].append((xName,numColumn))

                    tickPositions[nAxis] = catTickPositions
                    numericColumnPairs[nAxis] = tuple(columnNames)
                    axisLabels[nAxis] = {"x":categoricalColumns[1],"y":numColumn}
                    tickLabels[nAxis] = catTickLabels
                    axisLimits[nAxis] = {"yLimit":[globalMin-yMargin,globalMax+yMargin],"xLimit":[0-widthBox/2-border,nXAxisCats+widthBox/2+border]}
                    if not nAxisCat in axisTitles:
                        axisTitles[nAxisCat] = "{}:{}".format(categoricalColumns[2],axisCat)
                #saving internal id and color matches with pandas index (to allow easy manipulation)








            #tickPositions = dict([(n,np.arange(len(numericColumns))) for n in range(1)])
        #             index = data.sort_values(by=numColumn, ascending = self.indexSort == "ascending").index
        #            # index = data.index
        #             data = data.join(pd.Series(np.arange(index.size),index=index, name = indexName))
        #           #  print(kdeData)

        #  #kdeData, index = self.sourceData.getKernelDensity(dataID,pd.Series(numColumn),justData = True)
        #             #kdeData = [np.random.uniform(-x,x) for x in kdeData]
        #             #data.loc[index,indexName] = pd.Series(kdeData,index = index)


        return {"data":{"plotData":plotData,
            "axisPositions":axisPostions,
            "columnPairs":numericColumnPairs,
            "tickLabels": tickLabels,
            "tickPositions" : tickPositions,
            "axisTitles" : axisTitles,
            "dataColorGroups": colorGroupsData,
            "dataSizeGroups" : sizeGroupsData,
            "axisLimits" : axisLimits,
            #"dataSizeGroups" : sizeGroupsData,
            "axisLabels" : axisLabels,
            "subplotBorders":subplotBorders,
            "colorCategoryIndexMatch" : colorCategoryIndexMatch,
            "sizeCategoryIndexMatch" : colorCategoryIndexMatch,
            "multiScatterKwargs": multiScatterKwargs,
            "interalIDColumnPairs" : interalIDColumnPairs,
            "colorCategoricalColumn" : colorCategoricalColumn,
            "sizeCategoricalColumn" : colorCategoricalColumn,
            "dataID":dataID}}
