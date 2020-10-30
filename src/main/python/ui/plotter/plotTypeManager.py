

plotTypes = ["lineplot","pointplot","scatter","histogram","boxplot","barplot","violinplot","swarmplot","addSwarmplot","countplot","hclust","corrmatrix","x-ys-plot"]
fallbackType = "boxplot"
requiredProps = {
    "lineplot":{"numericalCols":2,"categoricalCols":0},
    "boxplot":{"numericalCols":1,"categoricalCols":0},
    "swarmplot":{"numericalCols":1,"categoricalCols":0},
    "addSwarmplot":{"numericalCols":1,"categoricalCols":0},
    "barplot":{"numericalCols":1,"categoricalCols":0},
    "scatter":{"numericalCols":2,"categoricalCols":0},
    "pointplot":{"numericalCols":1,"categoricalCols":0},
    "histogram":{"numericalCols":1,"categoricalCols":0},
    "time_series":{"numericalCols":2,"categoricalCols":0},
    "hclust":{"numericalCols":1,"categoricalCols":0},
    "corrmatrix":{"numericalCols":2,"categoricalCols":0},
    "violinplot":{"numericalCols":1,"categoricalCols":0},
    "countplot" : {"numericalCols":0,"categoricalCols":1},
    "x-ys-plot" : {"numericalCols":2,"categoricalCols":0}
}


plotTypeTooltips = {
    "lineplot":"Profile plot. Plots column data versus the row index.\nData are represented by the median (line with markers) and the first (0.25) and third (0.75) quantiles displayed as an area.",
    "pointplot":"Plots a pointplot showing the covidence interval by default. The error bar calculation can be defined in the context menu.",
    "scatter":"Scatter plot. Requires at least 2 numerical columns.",
    "histogram":"Density plot showing the distribution of numerical columns.\nAdding categorical columns will split the distributions. Several configuration options are available (Configuration)",
    "boxplot":"Plots a boxplot. Whiskers indicate min/max ignorng outliers.\nOutliers are ommitted if more than 20 boxes have to be plotted.",
    "violinplot":"Plots a violin plot using requires at least one numerical column.",
    "hclust":"Hierarchical clustering. Righ-click for several configurations such as colorbar range and label.\nHierarchical clustering can also be exported to excel.",
    "corrmatrix":"Correlation matrix using numerical columns ignoring categorical columns.\nBy default Pearson corelation is used.",
    "barplot":"Plots a barplot representing the mean. The error bar calculation can be defined in the context menu.",
    "swarmplot":"Swarmplot. Plot a swarm plot of the data.",
    "countplot":"Countplot. Plots the categorical group sizes in combination of a barchart and a lineplot indicating groups.",
    "addSwarmplot":"Swarmplot. Adds datapoints to selected plot types. Compatible charts are: boxplot, violin and barplot.",
    "x-ys-plot":"XYs Plot. Plots data of column vs column.\nYou can choose to have a common x-axis or to use every second column.\nOpen Settings to see more options."
}

gridPosition = {
        "lineplot":(0,0),
        "pointplot":(0,1),
        "scatter":(1,0),
        "x-ys-plot":(1,1),
        "time_series":(1,1),
        "histogram":(2,0),
        "barplot":(2,1),
        "boxplot":(3,0),
        "violinplot":(3,1),
        "swarmplot":(4,0),
        "addSwarmplot":(4,1),
        "hclust":(5,0),
        "corrmatrix":(5,1),
        "countplot":(6,0)
        }


warnColumns = "Number of numerical and categorical columns do not fit. Required columns: {} num. column(s) and {} categorical column(s). You provided: num: {} and cat. {}."

class PlotTypeManager(object):
    ""
    def __init__(self):
        ""
        self.plotTypes = plotTypes

    def isTypeValid(self,plotType,nNumCols,nCatCols):
        ""
        if plotType in requiredProps:
            reqProps = requiredProps[plotType]
            validInput = reqProps["numericalCols"] <= nNumCols and reqProps["categoricalCols"] <= nCatCols
            return validInput, "" if validInput else warnColumns.format(reqProps["numericalCols"],reqProps["categoricalCols"],nNumCols,nCatCols)
        return False,  "Plot Type unknown."

    def getDefaultType(self):
        ""
        return fallbackType

    def getAvailableTypes(self):
        ""
        return self.plotTypes