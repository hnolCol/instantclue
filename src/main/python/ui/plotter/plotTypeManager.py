

plotTypes = ["lineplot",
            "pointplot",
            "scatter",
            "mulitscatter",
            "histogram",
            "boxplot",
            "boxenplot",
            "barplot",
            "violinplot",
            "swarmplot",
            "addSwarmplot",
            "countplot",
            "hclust",
            "corrmatrix",
            "x-ys-plot",
            #"dim-red-plot",
            "clusterplot"]
            #"forestplot",
         #   "wordcloud"]
            
fallbackType = "boxplot"
requiredProps = {
    "lineplot":{"numericalCols":2,"categoricalCols":0},
    "boxplot":{"numericalCols":1,"categoricalCols":0},
    "boxenplot":{"numericalCols":1,"categoricalCols":0},
    "swarmplot":{"numericalCols":1,"categoricalCols":0},
    "addSwarmplot":{"numericalCols":1,"categoricalCols":0},
    "barplot":{"numericalCols":1,"categoricalCols":0},
    "scatter":{"numericalCols":1,"categoricalCols":0},
    "mulitscatter":{"numericalCols":2,"categoricalCols":0},
    "pointplot":{"numericalCols":1,"categoricalCols":0},
    "histogram":{"numericalCols":1,"categoricalCols":0},
    "time_series":{"numericalCols":2,"categoricalCols":0},
    "hclust":{"numericalCols":1,"categoricalCols":0},
    "corrmatrix":{"numericalCols":2,"categoricalCols":0},
    "violinplot":{"numericalCols":1,"categoricalCols":0},
    "countplot" : {"numericalCols":0,"categoricalCols":1},
    "x-ys-plot" : {"numericalCols":1,"categoricalCols":0},
    "dim-red-plot": {"numericalCols":2,"categoricalCols":0},
   # "forestplot" : {"numericalCols":1,"categoricalCols":1},
    #"wordcloud" : {"numericalCols":0,"categoricalCols":1},
    "clusterplot" : {"numericalCols":2,"categoricalCols":0}
}


plotTypeTooltips = {
    "lineplot":"Profile plot. Plots column data versus the row index.\nData are represented by the median (line with markers) and the first (0.25) and third (0.75) quantiles displayed as an area.",
    "pointplot":"Plots a pointplot showing the confidence interval by default. The error bar calculation can be defined in the context menu.",
    "scatter":"Scatter plot. Requires at least one numerical column (plotted against index)",
    "mulitscatter" : "Mutli-Scatter plot. Requires at least 2 numerical columns.",
    "histogram":"Density plot showing the distribution of numerical columns.\nAdding categorical columns will split the distributions. Several configuration options are available (Configuration)",
    "boxplot":"Plots a boxplot. Whiskers indicate min/max ignorng outliers.\nOutliers are ommitted if more than 20 boxes have to be plotted.",
    "violinplot":"Plots a violin plot using requires at least one numerical column.",
    "hclust":"Hierarchical clustering. Righ-click for several configurations such as colorbar range and label.\nHierarchical clustering can also be exported to excel.",
    "corrmatrix":"Correlation matrix using numerical columns ignoring categorical columns.\nBy default Pearson corelation is used.",
    "barplot":"Plots a barplot representing the mean. The error bar calculation can be defined in the context menu.",
    "boxenplot" : "Generates a boxenplot. Suitable to display distributions for large datasets.",
    "swarmplot":"Swarmplot. Plot a swarm plot of the data.",
    "countplot":"Countplot. Plots the categorical group sizes in combination of a barchart and a lineplot indicating groups.",
    "addSwarmplot":"Swarmplot. Adds datapoints to selected plot types. Compatible charts are: boxplot, violin and barplot.",
    "x-ys-plot":"XYs Plot. Plots data of column vs column.\nYou can choose to have a common x-axis or to use every second column.\nOpen Settings to see more options.",
    "dim-red-plot" : "Dimensional reduction.\nAvailable methods: PCA",
   # "forestplot"   : "Forest plot. Calculates odds ratios and confidence interval.",
    #"wordcloud"   : "Word Cloud - Use categorical column to create a word cloud.",
    "clusterplot" : "Clusterplot. Display clusters of data in multiple plot types (barplot/boxplot/lineplot)."
}

gridPosition = {
        "lineplot":(0,0),
        "pointplot":(0,1),
        "scatter":(1,0),
        "x-ys-plot":(1,1),
        
       # "time_series":(1,1),
        "mulitscatter":(2,0),
        "histogram":(3,0),
        "barplot":(3,1),
        "boxplot":(4,0),
        "boxenplot" : (4,1),
        "violinplot":(5,0),
        "swarmplot":(6,0),
        "addSwarmplot":(6,1),
        #"forestplot": (6,0),
        "hclust":(7,0),
        "corrmatrix":(7,1),
        "dim-red-plot" : (8,0),
        "clusterplot": (8,1),
        "countplot":(9,1),
       # "wordcloud" : (9,1)
        
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