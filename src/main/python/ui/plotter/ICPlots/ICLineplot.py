

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
import numpy as np

class ICLineplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICLineplot,self).__init__(*args,**kwargs)

        self.pointplotItems = dict() 

    def addHoverLine(self):
        ""
        self.hoverLines = {}
        for ax in self.axisDict.values():
            hoverLine = ax.plot([],[],
                            linewidth=self.getParam("linewidth.median"), 
                            marker= self.getParam("marker.median"), 
                            color = self.getParam("scatter.hover.color"),
                            markeredgecolor = "black", 
                            linestyle = "-",
                            markeredgewidth = self.getParam("markeredgewidth.median"))
            self.hoverLines[ax] = hoverLine[0]

        self.setHoverLinesInivisble()

    def setHoverLinesInivisble(self):
        ""
        for l in self.hoverLines.values():
            l.set_visible(False)

    def plotQuantiles(self,x,data,ax,color = "black",single=True, saveLines = True):
        ""
        if not data.shape[0] == 5:
            return
        q25 = data[1,:]
        q50 = data[2,:]
        q75 = data[3,:]
        #plot min and max quantiles
        #plot 25 and 75 quantiles
        qArea = ax.fill_between(x,q25,q75,alpha=self.getParam("alpha.IQR"),facecolor=color,edgecolors="black",linewidth=0.1)
        #plot median
        line = ax.plot(
                    x,
                    q50,
                    linestyle="-",
                    color="black" if single else color,
                    linewidth=self.getParam("linewidth.median"), 
                    marker= self.getParam("marker.median"),
                    markerfacecolor = color, 
                    markeredgecolor = "black", 
                    markeredgewidth = self.getParam("markeredgewidth.median"))
        if saveLines:
            internalID = self.getInternalIDByColor(color)
            if not internalID in self.lineplotItems:
                self.lineplotItems[internalID] = []
            self.lineplotItems[internalID].append(qArea)
            self.lineplotItems[internalID].append(line[0])


    def initLineplot(self, onlyForID = None, targetAx = None):
        ""

        try:
            if not hasattr(self,"lineplotItems"):
                self.lineplotItems = dict() 
            else:
                self.lineplotItems.clear() 

            for n, lineData in self.data["plotData"].items():                
                if n in self.axisDict:

                    if onlyForID is not None and n != onlyForID:
                        continue
                    elif onlyForID is not None and n == onlyForID and targetAx is not None:
                        ax = targetAx
                    else:
                        ax = self.axisDict[n]

                    singleLine = len(lineData) == 1 #will cause colorling the line in black
                    for q in lineData:
                        self.plotQuantiles(
                                        ax = ax, 
                                        data = q["quantiles"],
                                        x = q["xValues"],
                                        color= q["color"],
                                        single=singleLine,
                                        saveLines= onlyForID is None) #if this is None, we are not exporting to the main figure

                    
        except Exception as e:
            print(e)

    def onDataLoad(self, data):
        ""
        try:
            self.data = data
           
            self.initAxes(data["axisPositions"])
            self.initLineplot()
            for n,ax in self.axisDict.items():
                if n in self.data["axisLimits"]:
                    self.setAxisLimits(ax,
                            self.data["axisLimits"][n]["xLimit"],
                            self.data["axisLimits"][n]["yLimit"])

            if self.interactive:
               self.addHoverLine()
               self.addHoverBinding() 

            self.addTitles()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.setXTicksForAxes(self.axisDict,
                        data["tickPositions"],
                        data["tickLabels"],
                        rotation=90)
            qsData = self.getQuickSelectData()
            if qsData is not None:
                self.mC.quickSelectTrigger.emit()
            else:
                self.updateFigure.emit() 
           
           
        except Exception as e:
            print(e)
        

    def setHoverData(self,dataIndex, showText = False):
        ""
       # print(dataIndex)
       # if dataIndex in self.data["plotData"].index:
        
        dataIndex = dataIndex[0]
        for n,ax in self.axisDict.items():
            if not dataIndex in self.data["hoverData"][n].index:
                continue
            else:
                
                for axB in self.axBackground.keys():
                    self.p.f.canvas.restore_region(self.axBackground[axB])
                    
                y = self.data["hoverData"][n].loc[dataIndex,self.data["numericColumns"]].values
                x = np.arange(y.size)
                self.hoverLines[ax].set_visible(True)
                self.hoverLines[ax].set_data(x,y)
                ax.draw_artist(self.hoverLines[ax])
                break 
            
        #blit canvas
        self.p.f.canvas.blit(ax.bbox)


    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverLines"):
            for l in self.hoverLines.values():
                l.set_visible(False)

    def getInternalIDByColor(self, color):
        ""
        colorGroupData = self.data["dataColorGroups"]
        boolIdx = colorGroupData["color"].values ==  color
        if np.any(boolIdx):
            return colorGroupData.loc[boolIdx,"internalID"].values[0]

    def updateGroupColors(self,colorGroup,changedCategory=None):
        "changed category is encoded in a internalID"
      
        if self.colorCategoryIndexMatch is not None:
            
            if changedCategory in self.colorCategoryIndexMatch:
                indices = self.colorCategoryIndexMatch[changedCategory]

                for idx in indices:
                    if idx in self.quickSelectLines:
                        dataBool = colorGroup["internalID"] == changedCategory
                        c = colorGroup.loc[dataBool,"color"].values[0]
                        self.quickSelectLines[idx].set_color(c)
                        self.quickSelectLines[idx].set_markerfacecolor(c)
        else:

            for color, _ , internalID in colorGroup.values:
                if internalID in self.lineplotItems:
                    artits = self.lineplotItems[internalID]
                    for l in artits:
                        if hasattr(l,"set_facecolor"):
                            l.set_facecolor(color)
                        elif hasattr(l,"set_markerfacecolor"):
                            l.set_markerfacecolor(color)
                        if isinstance(l,Line2D): #2D line has marker colors and line colors
                            l.set_color(color)
                        
        if hasattr(self,"colorLegend"):
            self.addColorLegendToGraph(colorGroup,update=False)
        self.updateFigure.emit()

    def updateBackgrounds(self):
        "Update Background for blitting"
        self.axBackground = dict()
        for ax in self.axisDict.values():
            self.axBackground[ax] = self.p.f.canvas.copy_from_bbox(ax.bbox)	
    
    def updateQuickSelectItems(self,propsData=None):
        "Saves lines by idx id"

        colorData = self.getQuickSelectData()
        dataIndex = self.getDataIndexOfQuickSelectSelection()
        if not hasattr(self,"quickSelectLines"):
            self.quickSelectLines = dict() 
        #dataIndexInClust = [idx for idx in dataIndex if idx in self.data["plotData"].index]
        for n,ax in self.axisDict.items():
            idxInAxSet = self.data["hoverData"][n].index.intersection(dataIndex)
            if idxInAxSet.size > 0:
                for idx in idxInAxSet.values:

                    y = self.data["hoverData"][n].loc[idx,self.data["numericColumns"]].values.flatten()
                    x = np.arange(y.size)
                    c = colorData.loc[idx,"color"]
                    lines = ax.plot(
                                    x,
                                    y, 
                                    marker = self.getParam("marker.quickSelect"), 
                                    markerfacecolor = c, 
                                    color = c, 
                                    linewidth = self.getParam("linewidth.quickSelect"), 
                                    markeredgecolor = "black", 
                                    markeredgewidth = self.getParam("markeredgewidth.quickSelect")
                                    )
                    self.quickSelectLines[idx] = lines[0]
    
    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        
        data = self.data
        #self.setAxisLabels(self.axisDict,data["axisLabels"],onlyForID=axisID)
        self.initLineplot(axisID,targetAx)
        for n,ax in self.axisDict.items():
            if axisID == n and axisID in data["axisLimits"]:
                self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
    
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)                         
 #self.data = data
           
         