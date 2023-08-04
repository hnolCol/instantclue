

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd 
from typing import Iterable

class ICLineplot(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICLineplot,self).__init__(*args,**kwargs)

        self.quickSelectLineKwargs = {}
        self.quickSelectPolygonKwargs = {}
        self.quickSelectPolygon = {}
    
    def addGraphSpecActions(self,menus : dict) -> None:
        ""
        if "main" in menus and hasattr(menus["main"],"addAction"):
            menus["main"].addAction("Lineplot Style",lambda : self.mC.openSettings(specificSettingsTab ="Lineplot"))

    def addHoverLine(self):
        ""
        self.hoverLines = {}
        for ax in self.axisDict.values():
            hoverLine = ax.plot([],[],
                            linewidth=self.getParam("linewidth.median"), 
                            marker= None if  self.getParam("marker.median")  == "none" else self.getParam("marker.median"), 
                            color = self.getParam("scatter.hover.color"),
                            markeredgecolor = "black", 
                            linestyle = "-",
                            markeredgewidth = self.getParam("markeredgewidth.median"))
            self.hoverLines[ax] = hoverLine[0]

        self.setHoverLinesInivisble()

    def addHoverArea(self):
        ""
        self.hoverAreas = {}
        for ax in self.axisDict.values():
            hoverArea  = Polygon(
                            [[0,0],[1,1]],
                            visible=False,
                            alpha=self.getParam("alpha.IQR"),
                            facecolor=self.getParam("scatter.hover.color"),
                            edgecolor="black",
                            linewidth=0.1,
                            fill=True, 
                            closed=True)
            
            ax.add_patch(hoverArea)
            self.hoverAreas[ax] = hoverArea
        
        
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
                    marker= None if self.getParam("marker.median") == "none" else self.getParam("marker.median"),
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
               self.addHoverArea()
               self.addHoverBinding() 

            self.addTitles()
            self.setDataInColorTable(self.data["dataColorGroups"], title = self.data["colorCategoricalColumn"])
            self.setXTicksForAxes(self.axisDict,
                        data["tickPositions"],
                        data["tickLabels"],
                        rotation=90)
            self.checkForQuickSelectDataAndUpdateFigure()
           
           
        except Exception as e:
            print(e)
        

    def setHoverData(self,dataIndex : Iterable, showText : bool = False):
        ""
        for axB in self.axBackground.keys():
            self.p.f.canvas.restore_region(self.axBackground[axB])

        for n,ax in self.axisDict.items():

            idx = self.data["hoverData"][n].index.intersection(dataIndex)
            if not idx.empty:
                
                ys = self.data["hoverData"][n].loc[idx,self.data["numericColumns"]].values
                if ys.shape[0] > 1:
                    
                    columnQuantiles = np.nanquantile(ys, q = [0.25,0.5,0.75],axis=0)
                    ys = columnQuantiles[1,:]
                    q25 = columnQuantiles[0,:]
                    q75 = columnQuantiles[2,:]
                    xs = np.arange(ys.size)
                    polygonXY = np.array([(x,y) for x,y in zip(xs,q25)] + list(reversed([(x,y) for x,y in zip(xs,q75)])))
                    
                    self.hoverAreas[ax].set_xy(polygonXY)
                    self.hoverAreas[ax].set_visible(True)
                    ax.draw_artist(self.hoverAreas[ax])
                else:
                    xs = np.arange(ys.size)

                self.hoverLines[ax].set_visible(True)
                self.hoverLines[ax].set_data(xs,ys)
                

                ax.draw_artist(self.hoverLines[ax])
               
            
        #blit canvas
        self.p.f.canvas.blit(ax.bbox)


    def setHoverObjectsInvisible(self):
        ""
        if hasattr(self,"hoverLines"):
            for l in self.hoverLines.values():
                l.set_visible(False)
        if hasattr(self,"hoverAreas"):
            for area in self.hoverAreas.values():
                area.set_visible(False)

    def getInternalIDByColor(self, color : str):
        ""
        colorGroupData = self.data["dataColorGroups"]
        boolIdx = colorGroupData["color"].values ==  color
        if np.any(boolIdx):
            return colorGroupData.loc[boolIdx,"internalID"].values[0]

    def updateGroupColors(self,colorGroup : pd.DataFrame,changedCategory : str|None = None):
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

       # colorData = self.getQuickSelectData()
        
        if not hasattr(self,"quickSelectLines"):
            self.quickSelectLines = dict() 

        if not hasattr(self,"quickSelectAreas"):
            self.quickSelectAreas = dict() 

        # if self.isQuickSelectModeUnique() and hasattr(self,"quickSelectCategoryIndexMatch"):

        #     dataIndex = np.concatenate([idx for idx in self.quickSelectCategoryIndexMatch.values()])

        # else:

        #     dataIndex = self.getDataIndexOfQuickSelectSelection() 

        for intID, indics in self.quickSelectCategoryIndexMatch.items():
            for n,ax in self.axisDict.items():

                if ax not in self.quickSelectLines:
                    self.quickSelectLines[ax] = {}
                    self.quickSelectLineKwargs[ax] = {}
                if ax not in self.quickSelectPolygon:
                    self.quickSelectPolygonKwargs[ax] = {}
                    self.quickSelectPolygon[ax] = {}
                
                if intID in self.quickSelectLines[ax]:
                    continue

                c = propsData.loc[indics,"color"].values[0]

                idxInAxSet = self.data["hoverData"][n].index.intersection(indics)
                if idxInAxSet.size == 1:
                    y = self.data["hoverData"][n].loc[idxInAxSet,self.data["numericColumns"]].values.flatten()
                    x = np.arange(y.size)
                    lineKwargs = dict(
                                    xdata = x,
                                    ydata = y, 
                                    marker = self.getParam("marker.quickSelect"), 
                                    markerfacecolor = c, 
                                    color = c, 
                                    linewidth = self.getParam("linewidth.quickSelect"), 
                                    markeredgecolor = "black", 
                                    markeredgewidth = self.getParam("markeredgewidth.quickSelect")
                                    )
                    line = Line2D(**lineKwargs)
                    ax.add_line(line)
                    
                    self.quickSelectLineKwargs[ax][intID] = lineKwargs
                    self.quickSelectLines[ax][intID] = line
                
                elif idxInAxSet.size > 1:

                    ys = self.data["hoverData"][n].loc[idxInAxSet,self.data["numericColumns"]].values
                    columnQuantiles = np.nanquantile(ys, q = [0.25,0.5,0.75],axis=0)

                    ys = columnQuantiles[1,:]
                    q25 = columnQuantiles[0,:]
                    q75 = columnQuantiles[2,:]
                    xs = np.arange(ys.size)
                    polygonXY = np.array([(x,y) for x,y in zip(xs,q25)] + list(reversed([(x,y) for x,y in zip(xs,q75)])))
                    
                    
                    x = np.arange(ys.size)
                    lineKwargs = dict(
                                    xdata = xs,
                                    ydata = ys, 
                                    marker = self.getParam("marker.quickSelect"), 
                                    markerfacecolor = c, 
                                    color = c, 
                                    linewidth = self.getParam("linewidth.quickSelect"), 
                                    markeredgecolor = "black", 
                                    markeredgewidth = self.getParam("markeredgewidth.quickSelect")
                                    )
                    poylgonKwargs = dict(xy = polygonXY,
                                        visible=True,
                                        alpha=self.getParam("alpha.IQR"),
                                        facecolor=c,
                                        edgecolor="black",
                                        linewidth=0.1,
                                        fill=True, 
                                        closed=True)

                    line = Line2D(**lineKwargs)
                    ax.add_line(line)

                    poly = Polygon(**poylgonKwargs)
                    ax.add_patch(poly)
                    
                    self.quickSelectPolygonKwargs[ax][intID] = poylgonKwargs
                    self.quickSelectPolygon[ax][intID] = poly
                    self.quickSelectLineKwargs[ax][intID] = lineKwargs
                    self.quickSelectLines[ax][intID] = line


                else:
                    continue

                self.quickSelectScatterDataIdx[ax] = idxInAxSet.values
    
    def updateQuickSelectData(self,quickSelectGroup,changedCategory=None):
        ""
        c = quickSelectGroup.loc[quickSelectGroup["internalID"] == changedCategory]["color"].values[0]
        for ax in self.axisDict.values():
            if hasattr(self,"quickSelectLines") and ax in self.quickSelectLines and changedCategory in self.quickSelectLines[ax]:
                qSLine = self.quickSelectLines[ax][changedCategory]
                qSLine.set_color(c)
                qSLine.set_markerfacecolor(c)

            if hasattr(self,"quickSelectPolygon") and ax in self.quickSelectPolygon and changedCategory in self.quickSelectPolygon[ax]:

                    poly = self.quickSelectPolygon[ax][changedCategory]
                    poly.set_facecolor(c)

        self.updateFigure.emit()
    
    def mirrorQuickSelectArtists(self,axisID,targetAx):
        ""
        if axisID in self.axisDict:
            sourceAx = self.axisDict[axisID]
            if sourceAx in self.quickSelectLineKwargs:
                for lineKwargs in self.quickSelectLineKwargs[sourceAx].values():
                    targetAx.add_line(Line2D(**lineKwargs))
                for poylgonKwargs in self.quickSelectPolygonKwargs[sourceAx].values():
                    targetAx.add_patch(Polygon(**poylgonKwargs))

    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        
        data = self.data
        self.initLineplot(axisID,targetAx)
        for n,ax in self.axisDict.items():
            if axisID == n and axisID in data["axisLimits"]:
                self.setAxisLimits(ax,yLimit=data["axisLimits"][n]["yLimit"],xLimit=data["axisLimits"][n]["xLimit"])
    
        self.setXTicksForAxes({axisID:targetAx},data["tickPositions"],data["tickLabels"], onlyForID = axisID, rotation=90)                         
           

    def resetQuickSelectArtists(self):
        ""
        for ax in self.axisDict.values():
            if hasattr(self,"quickSelectPolygon") and ax in self.quickSelectPolygon:
                for poly in self.quickSelectPolygon[ax].values():
                    poly.set_visible(False)
            if hasattr(self,"quickSelectLines") and ax in self.quickSelectLines:
                for line in self.quickSelectLines[ax].values():
                    line.set_visible(False)

        self.quickSelectPolygonKwargs.clear()
        self.quickSelectLineKwargs.clear()

        if hasattr(self,"quickSelectPolygon"):
            self.quickSelectPolygon.clear()
        if hasattr(self,"quickSelectLines"):
            self.quickSelectLines.clear()
            
        self.updateFigure.emit()