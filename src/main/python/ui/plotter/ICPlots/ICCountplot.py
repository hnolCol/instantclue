

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from typing import Iterable
import numpy as np

class ICCountplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICCountplot,self).__init__(*args,**kwargs)

    def addGraphSpecActions(self,menus : dict) -> None:
        ""
        menus["main"].addAction("Show subset", self.openSubsetDataInDialog)
        menus["main"].addAction("Annotate subset.", self.annotateSubsetInData)
        menus["main"].addAction("Show count data", self.openCountDataInDialog)


    def addHoverLine(self):
        ""
        lKwargs = self.getLineKwargs()
        lKwargs["color"] = self.getParam("scatter.hover.color")
        lKwargs["markerfacecolor"] = self.getParam("scatter.hover.color")
        self.hoverLine = Line2D(xdata=[],ydata=[],**lKwargs)
        self.axisDict[1].add_artist(self.hoverLine)

    def addHoverBar(self):
        ""
        self.hoverRectangle = Rectangle(xy = (-100,0), width = 1, height = 0)
        self.hoverRectangle.set_visible(False)
        self.axisDict[0].add_artist(self.hoverRectangle)

        self.hoverRectangle2 = Rectangle(xy = (-100,0), width = 1, height = 0)
        self.hoverRectangle2.set_visible(False)
        self.axisDict[2].add_artist(self.hoverRectangle2)

    def addLabelsToBar(self,rects,ax,dataKey = "rawCounts", horizontal = True):
        ""
        if self.getParam("show.counts"):
            for n,rect in enumerate(rects):
                height = rect.get_height()
                txt = self.data[dataKey][n]
                if horizontal:
                    ax.annotate(
                        txt,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontproperties = FontProperties(
                                            family=self.getParam("annotationFontFamily"),
                                            size = self.getParam("annotationFontSize"))
                            )
                else:
                    ax.annotate(
                        txt,
                        xy=(rect.get_width(), rect.get_y() + rect.get_height()/2),
                        xytext=(3, 0),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='left', va='center',
                        fontproperties = FontProperties(
                                            family=self.getParam("annotationFontFamily"),
                                            size = self.getParam("annotationFontSize"))
                            )

    def addIndicatorBars(self):
        ""
        numAreas = self.data["axisLimits"][1]["yLimit"][1] + 1
        width = self.data["axisLimits"][1]["xLimit"][1] + 0.5
        height = 1

        for n in range(numAreas):
            r = Rectangle(
                        xy = (-0.5,n-0.5), 
                        width = width, 
                        height = height, 
                        fill=True, 
                        linewidth = 0.5,
                        edgecolor = "black",
                        facecolor = "#f2f2f2" if n % 2 == 0 else "#c9c9c9")
            self.axisDict[1].add_patch(r)


    def annotateSubsetInData(self):
        "Annotate the features behind a categorical count plot in the source data."
        if hasattr(self,"idx") and self.idx is not None and "hoverData" in self.data and "dataID" in self.data: 
            dataIndex = self.data["hoverData"][self.idx]
            categories = self.data["chartData"].iloc[self.idx].loc[[columnName for columnName in self.data["chartData"].columns if columnName != "counts"]]
            dataID = self.data["dataID"]
            columnNameProps = [f"{columName}: {categoricalValue}" for columName,categoricalValue in zip(categories.index,categories.values)]
            funcProps = {
                "key" : "data::annotateDataByIndicies",
                "kwargs" : {
                    "dataID" : dataID,
                    "indices" : dataIndex,
                    "columnName" : " ".join(columnNameProps),

                }
            }

            self.mC.sendRequestToThread(funcProps)


    def getLineKwargs(self):
        ""
        lineKwargs = {

            "marker" : self.getParam("counts.marker"),
            "markersize" : self.getParam("counts.markersize"),
            "markerfacecolor" : self.getParam("counts.markerfacecolor"),
            "markeredgecolor" : self.getParam("counts.markeredgecolor"),
            "linestyle" : self.getParam("counts.linestyle"),
            "linewidth" : self.getParam("counts.linewidth"),
            "color" : self.getParam("counts.linecolor")
        }
        return lineKwargs

    def initBarplot(self):
        ""

        self.rects = self.axisDict[0].bar(**self.data["plotData"]["bar-counts"])
        self.rects2 = self.axisDict[2].barh(**self.data["plotData"]["total-bar-count"])
        
        

    def initLineplot(self):
        ""
        self.lineplotItems = dict() 
        for idx,lData in self.data["plotData"]["lineplot"].items():
            l = Line2D(**lData,**self.getLineKwargs())
            self.lineplotItems[idx] = l
            self.axisDict[1].add_artist(l)

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
            self.initAxes(data["axisPositions"])
            self.addIndicatorBars() 
            self.initLineplot()
            self.initBarplot()
            self.addLabelsToBar(self.rects,self.axisDict[0])
            self.addLabelsToBar(self.rects2,self.axisDict[2],dataKey="rawTotalCounts",horizontal=False)
            self.setXTicks(self.axisDict[0],[],[])
            self.setYTicks( self.axisDict[2],[],[])
            if self.interactive:
                self.addHoverBinding()
                self.addHoverLine()
                self.addHoverBar()

            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])
            self.setYTicks(
                    self.axisDict[1],
                    self.data["tickPositions"][1]["y"],
                    self.data["tickLabels"][1]["y"]
                    )
            self.setXTicks(
                    self.axisDict[1],
                    self.data["tickPositions"][1]["x"],
                    self.data["tickLabels"][1]["x"]
                    )
            self.axisDict[0].set_ylabel("Counts" if self.getParam("countTransform") == "none" else "Counts ({})".format(self.getParam("countTransform")))
            self.axisDict[1].set_xlabel("Categorical Groups")
            self.axisDict[2].set_xlabel("Total Categorical Counts")

            self.coloredTicks = dict() 
            for ytick, color in zip(self.axisDict[1].get_yticklabels(), self.data["tickColors"]):
                internalID = self.getInternalIDByColor(color)
                if internalID not in self.coloredTicks:
                    self.coloredTicks[internalID] = []
                self.coloredTicks[internalID].append(ytick)
                ytick.set_color(color)
                ytick.set_fontweight("bold")

            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.updateFigure.emit() 
           
           
        except Exception as e:
            print(e)
        

    def onHover(self,event):
        ""

        if event.inaxes is None:
           # self.setHoverObjectsInvisible()
            #self.drawBackgrounds()
            self.idx = None
            return

        ax = event.inaxes
        
        if ax == self.axisDict[0]:
            itemIdx = [n for n,rect in enumerate(self.rects) if rect.contains(event)[0]]                
        else:
            itemIdx = [idx for idx,line in self.lineplotItems.items() if line.contains(event)[0]]
            
        if len(itemIdx) == 0:
            self.setHoverObjectsInvisible()
            self.drawBackgrounds()
            self.idx = None
            return
        self.idx = itemIdx[0]
        
        self.updateHoverLineData(self.axisDict[1],self.idx )
        self.updateHoverBar(self.axisDict[0],self.idx )
        
        if isinstance(event.ydata,float) and event.inaxes == self.axisDict[1]: #if hover over bottom (scatter ax) -> show hover in right bar graph.
            self.updateHoverBar(self.axisDict[2],int(event.ydata+0.5),self.rects2)

        dataIndex = self.data["hoverData"][self.idx]
        
        if self.isQuickSelectActive():
            self.sendIndexToQuickSelectWidget(dataIndex)
        if self.isLiveGraphActive():
            self.sendIndexToLiveGraph(dataIndex)

    def openSubsetDataInDialog(self):
        ""
        if self.idx is not None:
            
            dataIndex = self.data["hoverData"][self.idx]
            dataID = self.data["dataID"]
            columnNames = self.mC.data.getPlainColumnNames(dataID)
            dataToShow = self.mC.data.getDataByColumnNames(dataID,columnNames,ignore_clipping=False)["fnKwargs"]["data"].loc[dataIndex,:]
            self.mC.mainFrames["data"].openDataFrameinDialog(dataToShow,
                                                            ignoreChanges=True, 
                                                            headerLabel="Data in countplot", 
                                                            )
        else:
            self.mC.sendToWarningDialog(infoText="Please hover over a certain group (indicated by red coloring of the bars) to select a subset first.")

    def openCountDataInDialog(self):
        ""
        if "chartData" in self.data and hasattr(self.mC,"mainFrames"):
            self.mC.mainFrames["data"].openDataFrameinDialog(self.data["chartData"],
                                                            ignoreChanges=True, 
                                                            headerLabel="Count plot data.", 
                                                            tableKwargs={"forwardSelectionToGraph":False})
        
    def updateHoverLineData(self,ax, idx):
        ""
        self.p.f.canvas.restore_region(self.axBackground[ax])
        data = self.lineplotItems[idx].get_xydata() 
        self.hoverLine.set_data(data[:,0],data[:,1])
        self.hoverLine.set_visible(True)
        ax.draw_artist(self.hoverLine)
        self.p.f.canvas.blit(ax.bbox)

    def updateHoverBar(self,ax,idx,rects=None):
        ""
        if rects is None:
            rect = self.rects[idx]
            hoverRectangle = self.hoverRectangle
        else:
            #rects is given -> can be applied to any ax.
            rect = rects[idx]   
            hoverRectangle = self.hoverRectangle2    
    # self.hoverRectangle.update_from(rect)
        self.p.f.canvas.restore_region(self.axBackground[ax])
        hoverRectangle.set_facecolor(self.getParam("scatter.hover.color"))
        hoverRectangle.set_xy(rect.get_xy())
        hoverRectangle.set_width(rect.get_width())
        hoverRectangle.set_height(rect.get_height())
        hoverRectangle.set_visible(True)
        ax.draw_artist(hoverRectangle)
        self.p.f.canvas.blit(ax.bbox)

    def setHoverData(self,dataIndex : Iterable, showText = False):
        ""
        idxHover = []

        ax = self.axisDict[1]
        if ax in self.axBackground:
            idxHover = [idx for idx,data in self.data["hoverData"].items() if any(x in data for x in dataIndex)]
            if len(idxHover) > 0:
                self.updateHoverLineData(ax,idxHover[0])

        ax = self.axisDict[0]
        
        if ax in self.axBackground:
            
            if len(idxHover) > 0:
                self.updateHoverBar(ax,idxHover[0])
                 
            else:
                self.setHoverObjectsInvisible()
            
    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverLine"):
            self.hoverLine.set_visible(False)
           
        if hasattr(self,"hoverRectangle"):
            self.hoverRectangle.set_visible(False)
        if hasattr(self,"hoverRectangle2"):
            self.hoverRectangle2.set_visible(False)
            
    def getInternalIDByColor(self, color):
        ""
        colorGroupData = self.data["dataColorGroups"]
        boolIdx = colorGroupData["color"].values ==  color
        if np.any(boolIdx):
            return colorGroupData.loc[boolIdx,"internalID"].values[0]

    def updateGroupColors(self,colorGroup,changedCategory=None):
        "changed category is encoded in a internalID"
        for color, _ , internalID in colorGroup.values:
            if internalID in self.coloredTicks:
                artits = self.coloredTicks[internalID]
                for l in artits:
                    l.set_color(color)

                    # boolIdx = colorGroup["internalID"].values ==  changedCategory
                        #newColor = colorGroup.loc[boolIdx,"color"].values[0]
                        #l.set_markerfacecolor(color)
        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()

    def updateBackgrounds(self):
        "Update Background for blitting"
        self.axBackground = dict()
        for ax in self.axisDict.values():
            self.axBackground[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)	

    def drawBackgrounds(self):
        ""
        for ax, background in self.axBackground.items():
            self.p.f.canvas.restore_region(background)
            self.p.f.canvas.blit(ax.bbox)

    def updateQuickSelectItems(self):
        "Saves lines by idx id"

        # colorData = self.getQuickSelectData()
        # dataIndex = self.getDataIndexOfQuickSelectSelection()
        # if not hasattr(self,"quickSelectLines"):
        #     self.quickSelectLines = dict() 
        # #dataIndexInClust = [idx for idx in dataIndex if idx in self.data["plotData"].index]
        # for n,ax in self.axisDict.items():
        #     idxInAxSet = self.data["hoverData"][n].index.intersection(dataIndex)
        #     if idxInAxSet.size > 0:
        #         for idx in idxInAxSet.values:

        #             y = self.data["hoverData"][n].loc[idx,self.data["numericColumns"]].values.flatten()
        #             x = np.arange(y.size)
        #             c = colorData.loc[idx,"color"]
        #             lines = ax.plot(
        #                             x,
        #                             y, 
        #                             marker = self.getParam("marker.quickSelect"), 
        #                             markerfacecolor = c, 
        #                             color = c, 
        #                             linewidth = self.getParam("linewidth.quickSelect"), 
        #                             markeredgecolor = "black", 
        #                             markeredgewidth = self.getParam("markeredgewidth.quickSelect")
        #                             )
        #             self.quickSelectLines[idx] = lines[0]
                    
