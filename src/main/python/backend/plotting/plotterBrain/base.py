from .helpers import *
from .helpers import _width_functions, _lv_box_ends, _lv_outliers

class PlotterBrainBase(object):
    """Shared state, dispatch and generic plot helpers."""

    def __init__(self, sourceData):
        ""
        self.sourceData = sourceData
        self.axisBoxColor = "#F0F0F0"
        self.boxplotCapsLineWidth = 0.5
        self.axisBorderLineWidth = 1
        self.axisLineWidth = 1
        self.corrMatrixMethod = "pearson"
        self.maxColumns = 4
        self.colorColumn = None
        self.colorColumnType = "Categories"
        self.sizeColumn = None
        self.sizeColumnType = "Categories"
        self.scatterSize = 50
        self.maxScatterSize = 120
        self.minScatterSize = 20
        self.aggMethod = "mean"
        self.plotAgainstIndex = False
        self.indexSort = "ascending"
        self.barplotError = "CI (95%)"
        self.barplotMetric = "mean"
        self.histogramBins = 15
        self.histogramCumulative = False
        self.histogramDensity = False
        self.histogramHisttype = "bar"
        self.histogramLog = False
        self.histogramLinewidth = 1
        self.histogramSortCategories = False
        self.countTransform = "none"

    def figToClipboard(self,figure):
        ""
       # print(figure)
        buf = io.BytesIO()
        #pickle.dump(figure,buf)
        dpi = self.sourceData.parent.config.getParam("copy.to.clipboard.dpi")
        transparent = self.sourceData.parent.config.getParam("copy.to.clipboard.background.transparent")
        figure.savefig(buf,format="png",dpi=dpi,transparent=transparent)
        funcProps = getMessageProps("Done..","Figure saved to clipboard.")
        funcProps["buf"] = buf
        return funcProps

    def findPositions(self,numericColumns,categoricalColumns, plotType):
        ""
        nNumCol = len(numericColumns)
        if len(categoricalColumns) == 0:

            if plotType == "boxplot":
                width = 0.8
                x0 = np.linspace(0,nNumCol-1, num = nNumCol) - width/2
                return x0,width

    def getPlotProps(self, dataID, numericColumns, categoricalColumns = None, plotType = "scatter", **kwargs):
        ""

        self.numericColumns = numericColumns
        self.categoricalColumns = categoricalColumns
        self.plotType = plotType
        with threadpool_limits(limits=1, user_api='blas'):

            graphData = getattr(self,plotFnDict[plotType])(dataID,numericColumns,categoricalColumns,**kwargs)
            graphData["plotType"] = plotType
            return graphData

    def getCountplotProps(self,dataID, numericColumns, categoricalColumns,*args,**kwargs):
        ""
        colorGroups = pd.DataFrame(columns = ["color","group","internalID"])
        data = self.sourceData.getDataByColumnNames(dataID,numericColumns + categoricalColumns)["fnKwargs"]["data"]
        subplotBorders = dict(wspace=0.05, hspace = 0.0,bottom=0.15,right=0.95,top=0.95)


        if len(numericColumns) > 0 and len(categoricalColumns) == 0:
            print("numeric count plot")
            #visualize missing data using the countplot

            return getMessageProps("Error..","Not yet implemented.")



        axisDict = OrderedDict([(0,[2,4,(1,3)]),(1,[2,4,(5,7)]),(2,[2,4,8])]) #getAxisPosition(3, maxCol = 1)
        groupbyCatColumns = data.groupby(by=categoricalColumns, sort=False)
        colors = self.sourceData.colorManager.getNColorsByCurrentColorMap(len(categoricalColumns),"countplotLabelColorMap")
        groupSizes = groupbyCatColumns.size().sort_values(ascending=False).reset_index(name='counts')
        totalCountData = []
        for categoricalColumn in categoricalColumns:
            for groupName, groupData in groupSizes.groupby(by=categoricalColumn,sort=False):
                c = groupData["counts"].sum()
                totalCountData.append((groupName,c))

        if groupbyCatColumns.ngroups > 30:
            return getMessageProps("Error..","More than 30 unique categories found. This plot type is not appropiate for so many categories.")

        colorGroups["group"] = categoricalColumns
        colorGroups["color"] = colors
        colorGroups["internalID"] = [getRandomString() for _ in categoricalColumns]
        uniqueValues = OrderedDict([(catColumn, groupSizes[catColumn].unique()) for catColumn in categoricalColumns])
        tickColors = np.concatenate([[colors[n]] * v.size for n,v in enumerate(uniqueValues.values())])

        numUniqueValues = np.sum([v.size for v in uniqueValues.values()])
        uniqueValueList = np.concatenate([v for v in uniqueValues.values()])
        xLimitMax = groupSizes.index.size
        xCountValues = np.arange(groupSizes.index.size)
        rawCounts = groupSizes["counts"]
        if self.countTransform != "none" and self.countTransform in transformDict:
            groupSizes["counts"] = transformDict[self.countTransform](rawCounts)
        yCountValues = groupSizes["counts"]
        maxCountValue = np.max(yCountValues)
        yTicksPoints = np.arange(numUniqueValues)
        #get facotrs for unique cats
        factors = OrderedDict()
        for n, catCol in enumerate(categoricalColumns):
            codes, _ = groupSizes[catCol].factorize()
            if n == 0:
                factors[catCol] = codes
            else:
                maxFactors = np.max([v for v in factors.values()]) + 1
                codes = codes + maxFactors
                factors[catCol] = codes
        lines = {}
        hoverData = {}
        for idx in groupSizes.index:
            groupValues = groupSizes.loc[idx,categoricalColumns].values
            if len(categoricalColumns) > 1:

                hoverData[idx] = groupbyCatColumns.get_group(tuple(groupValues)).index
            else:
                hoverData[idx] = groupbyCatColumns.get_group(groupValues[0]).index

            vs = np.array([(idx, factors[categoricalColumns[n]][idx]) for n,x in enumerate(groupValues)])
            lines[idx] = {"xdata" :vs[:,0],"ydata" : vs[:,1]}

        return {"data":{
            "dataID" : dataID,
            "axisPositions":axisDict,
            "subplotBorders":subplotBorders,
            "rawCounts" : rawCounts,
            "rawTotalCounts" : [x[1] for x in totalCountData],
            "tickColors" : tickColors,
            "plotData" : {
                    "bar-counts" :
                            {"x":xCountValues,"height" : yCountValues},
                    "total-bar-count" :
                            {"width" : [x[1] for x in totalCountData], "y" : np.arange(len(totalCountData))},
                    "lineplot": lines},
            "axisLimits" : {0:{"xLimit" : [-0.5,xLimitMax-0.5], "yLimit" : [0,maxCountValue+0.01*maxCountValue]},
                            1:{"xLimit" : [-0.5,xLimitMax-0.5], "yLimit" : [-0.5,numUniqueValues]},
                            2:{"xLimit" : None, "yLimit" : [-0.5,numUniqueValues]}
                            },
            "tickPositions" : {1:{"y":yTicksPoints,"x":np.arange(xLimitMax)}},
            "tickLabels" : {1:{"y":uniqueValueList,"x": [""] * xLimitMax}},
            "barLabels" : yCountValues,
            "hoverData" : hoverData,
            "colorCategoricalColumn" : "Categorical Columns",
            "dataColorGroups": colorGroups,
            "chartData" : groupSizes


        }}

    def getNumericColorMappingGroupNames(self, maxV,q75,median,q25,minV):
        ""
        groupNames = ["Max ({})".format(getReadableNumber(maxV)),
                            "75% Quantile ({})".format(getReadableNumber(q75)),
                            "Median ({})".format(getReadableNumber(median)),
                            "25% Quantile ({})".format(getReadableNumber(q25)),
                            "Min ({})".format(getReadableNumber(minV))]
        return groupNames

    def getColorMapDictAndLayers(self, colorCategories, N):
        ""
        colors, layerMap = self.sourceData.colorManager.createColorMapDict(colorCategories,
                                                                        addNaNLevels=[(self.sourceData.replaceObjectNan,)*N],
                                                                        as_hex=True)
        return colors,layerMap

    def getColorMapper(self,uniqueValues):
        ""
        colorMap, _ = self.sourceData.colorManager.createColorMapDict(uniqueValues, as_hex=True)
        return colorMap

    def exportFigure(self,filePath, exporter):
        ""
        try:
            exporter.export(filePath)
            completeKwargs = getMessageProps("Exported..","Figure saved.")
        except Exception as e:
            print(e)

        return completeKwargs

    def _findScatterSubplotProps(self,numericPairs):
        ""
        subplotProps = dict(wspace=0.15, hspace = 0.15,bottom=0.15,right=0.95,top=0.95)
        n_pairs = len(numericPairs)
        if  n_pairs == 1:
            return 1,1,subplotProps

        subplotProps["wspace"]=0.30
        subplotProps["hspace"]=0.30

        if n_pairs < 5:
            return 2,2, subplotProps
        else:
            rows = int(np.ceil(n_pairs/3))
            columns = 3
            return rows, columns, subplotProps
