

from .ICChart import ICChart
from collections import OrderedDict
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
import numpy as np

class ICWordCloud(ICChart):
    ""
    def __init__(self,*args,**kwargs):
        ""
        super(ICWordCloud,self).__init__(*args,**kwargs)
        self.reqKeywords = ["cloud","axisPositions"]
      
    def initWordCloud(self):
        ""

        self.axisDict[0].imshow(self.data["cloud"], interpolation="bilinear")
        

    def onDataLoad(self, data):
        ""
        if all(reqKeyword in data for reqKeyword in self.reqKeywords):
            self.data = data
            self.initAxes(data["axisPositions"])
            self.setAxisOff(self.axisDict[0])
            self.initWordCloud()
            self.updateFigure.emit()
    
    
    
    def mirrorAxisContent(self, axisID, targetAx,*args,**kwargs):
        ""
        data = self.data 
        targetAx.imshow(data["cloud"], interpolation="bilinear")
        self.setAxisOff(targetAx)