import numpy as np
import pandas as pd

from collections import OrderedDict

from scipy.misc import comb
import seaborn as sns

import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource, LinearColorMapper,ColorBar,CDSView,BooleanFilter
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import viridis,GnBu,brewer, magma
from bokeh.layouts import column, gridplot, row, layout
from bokeh.transform import transform
from bokeh.models.glyphs import Patches
from bokeh.models import HoverTool,BoxSelectTool,TapTool
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
from bokeh.models.glyphs import MultiLine



legendAttr = dict(location="top_right",
           click_policy="hide",
           background_fill_color="#efefef",
           background_fill_alpha=0.75,
           border_line_color="darkgrey",
           label_text_color="darkgrey",
           label_text_font_size="8pt")

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def pol2cartY(rho, phi):
    #x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return y

class choordCoordinates(object):


  def __init__(self, data, format = "matrix",
               opacityIdo = 0.5,
               opacityChord = 0.3,
               lineColor = "black",
               gapFraction = 0.005,
               innerRadius = .93,
               choordRadius = 0.88,
               minTickLevels = 5,
               majTickLevels = 10,
               colorPalette = 'Blues',
               polarCoords = False,
               plotEngine = "matplotlib",
               lim = 1.3):
    ""
    self.data = data
    self.format = format
    self.gapFraction = gapFraction


    #inner and outer idogram limits
    self.outer = 1
    self.inner = innerRadius
    self.choordRad = choordRadius
    self.minTicks = minTickLevels
    self.majTicks = majTickLevels
    self.colorPalette = colorPalette
    self.polarCoords = polarCoords
    self.plotEngine = plotEngine
    self.lim = lim


    if self.checkData():
      print('data checked...')
      self.defineColors()
      self.createIdogram()


  def defineColors(self):
    ""
    self.colors = dict()
    colors = sns.color_palette(self.colorPalette, len(self.data.index)).as_hex()
    for name, col in zip(self.data.columns,colors):
      self.colors[name] = col


  def checkData(self):
    ""

    if self.format == "matrix":
      if len(self.data.index) != self.data.columns.size:
        return False

    return True


  def createIdogram(self):
    ""
    self.calculatePolarCoords()
    self.saveSectors()
    self.createChoords()


  def createChoords(self):
    ""
    self.lastCoord = OrderedDict()
    self.interactionCoords = OrderedDict()
    self.choords = OrderedDict()

   #to = self.data.columns.values.tolist()
    n = 1
    for startName in self.data.columns:
      x1Start, x2Start = self.sectors[startName]
      xDelta = x2Start - x1Start
      self.choords[startName] = []

      for n , (name, coords) in enumerate(self.sectors.items()):

      #get limits
        numInts =  self.data.loc[startName,name]
        if numInts == 0:          	
          pass
        else:
          fracInts = numInts/self.totalInts[startName]

          fracOfSector = xDelta * fracInts

          if n == 0 and startName not in self.lastCoord:
            c = (x1Start,x1Start+fracOfSector)
          else:
            if startName in self.lastCoord:
            	x1,x2 = self.lastCoord[startName] 
            else:
            	x2 = 0

          	
          		           	            
            c = (x2,x2+fracOfSector)

          self.lastCoord[startName] = c

          x1R, x2R = coords
         # deltaR = x2R - x1R

          if name in self.lastCoord:
            _,x2End = self.lastCoord[name]
          else:
            x2End = x1R

          fracIntR = numInts/self.totalInts[name]
          fracSectorR = (x2R - x1R) * fracIntR
          cR = (x2End, x2End + fracSectorR)

          self.lastCoord[name] = cR

          lineCoords = self.bezierCurve(self.transformToCartesian(np.array([self.choordRad] * 2),
                                          np.array([c[0],cR[1]])))

          lineCoords1 = self.bezierCurve(self.transformToCartesian(np.array([self.choordRad]*2),
                                          np.array([c[1],cR[0]])))

          if self.polarCoords:

            theta1, r1 = self.toPolar(lineCoords[0],lineCoords[1])
            theta2, r2 = self.toPolar(lineCoords1[0],lineCoords1[1])

            self.choords[startName].append([theta1,r1,theta2,r2])

          else:

            self.choords[startName].append([lineCoords, lineCoords1])



  def calculatePolarCoords(self):
    ""
    self.totalInts = OrderedDict()

    for name in self.data.columns:
      self.totalInts[name] = np.sum( self.data.loc[name,].values) + np.sum(self.data.loc[:,name].values)
    groupSize = list(self.totalInts.values())

    cumTotal = np.cumsum(groupSize)/ np.sum(groupSize)
    gapSize = self.gapFraction/2 * 2 * np.pi
    self.idoCoords = OrderedDict()

    for n,x in enumerate(cumTotal):

      if n == 0:

        x1 = 0 + gapSize
        x2 = x * 2 * np.pi - gapSize

      else:

        x1 = self.idoCoords[n-1][-1] + 2 * gapSize
        x2 = x * 2 * np.pi - gapSize

      self.idoCoords[n] = (x1,x2)


  def saveSectors(self):
      ""

      self.sectors = OrderedDict()
      ids = self.data.columns
      for n,name in enumerate(ids):
        self.sectors[name] = self.idoCoords[n]


  def addTicks(self,x1,x2, name, ax):
    ""
    totalInt = self.totalInts[name]


    numMinTicks = int(round(totalInt / self.minTicks , 0))

    startT = 0
    ticks = [0+self.minTicks*n for n in range(numMinTicks)]
    theta = np.linspace(x1,x2,num=numMinTicks)

    # calculate radii
    outerR = self.outer+0.02*self.outer
    outerRmin = self.outer+0.01*self.outer
    outerRText = outerR+0.02*self.outer

    for n,(t,textString) in enumerate(zip(theta,ticks)):

      if n == 0 or n % self.majTicks == 0:

        ax.plot([t,t],[self.outer,outerR], 'grey', linewidth = 0.5)

        rot = np.degrees(t)

        if rot > 90 and rot < 270:
          ha = 'right'
          rot += 180
        else:
          ha = 'left'

        ax.text(t,
              outerRText,
              textString,
              ha = ha,
              va = 'center',
              rotation = rot,
              rotation_mode = 'anchor',
              fontdict = {'family':'serif','size':6})

      else:
        ax.plot([t,t],[self.outer,outerRmin], 'grey', linewidth = 0.3)



  def idogramInCartesian(self, nTimes = 500):
    "Replcae polar with caertesian line coordinates"
    idoChordsCart = OrderedDict()
    innerR = np.full(nTimes,self.inner)
    outerR = np.full(nTimes,self.outer)
    for idx, (x1,x2) in self.idoCoords.items():
      phi = np.linspace(x1,x2,num=nTimes)

      rS = np.linspace(self.inner,self.outer,num=int(nTimes/20))
      phi1 = np.full(int(nTimes/20),x1)
      phi2 = np.full(int(nTimes/20),x2)
      x = np.concatenate([innerR * np.cos(phi),  #inner line
                          rS * np.cos(phi2),     #connect inner & outer line
                          outerR * np.cos(np.flip(phi)), # outer line
                          np.flip(rS) * np.cos(phi1)]) # connect outer & inner line

      y = np.concatenate([innerR * np.sin(phi),
                          rS * np.sin(phi2),
                          outerR * np.sin(np.flip(phi)),
                          np.flip(rS) * np.sin(phi1)])

      idoChordsCart[idx] = [x,y]

    self.idoCoords = idoChordsCart

  def arctan2abs(self,y1,x1):

    phi = np.arctan2(y1,x1)
    if phi > 0:
        return phi
    else:
        return 2*np.pi - np.abs(phi)


  def connectChoordsInCartesian(self, inOutlines, nTimes = 100):
    ""
    innerLine, outerLine = inOutlines

    xLine1, yLine1 = innerLine
    xLine2, yLine2 = outerLine


    #connectLines - 1st
    x1End, y1End = xLine1[-1], yLine1[-1]

    #connectLines - 2nd
    x2End, y2End = xLine2[-1], yLine2[-1]

    rInner = np.sqrt(x1End**2 + y1End**2) # is constant
    rS = np.full(nTimes,rInner)

    phiLine1 = self.arctan2abs(y1End,x1End)
    phiLine2 = self.arctan2abs(y2End,x2End)
    phi1 = np.linspace(phiLine1,phiLine2,num=nTimes)  if phiLine1 > phiLine2 else np.linspace(phiLine2,phiLine1,num=nTimes)

    x1Start, y1Start = xLine1[0],yLine1[0]
    x2Start, y2Start = xLine2[0],yLine2[0]
    phiLine1 = self.arctan2abs(y1Start,x1Start)
    phiLine2 = self.arctan2abs(y2Start,x2Start)

    phi2 = np.linspace(phiLine1,phiLine2,num=nTimes) if phiLine1 > phiLine2 else np.linspace(phiLine2,phiLine1,num=nTimes)

    x = np.concatenate([xLine2,
                      rS * np.cos(phi1),
                       np.flip(xLine1),
                       rS * np.cos(phi2)])

    y = np.concatenate([yLine2,
                        rS * np.sin(phi1),
                       np.flip(yLine1),
                        rS * np.sin(phi2)
                       ])

    return x,y



  def plotBokeh(self, figure = None):

    if self.polarCoords:

      raise Exception("Polar coordinates are only available for matplotlib, not bokeh, set plotEngine to 'matplotlib'.")

    else:

      self.CDS = OrderedDict()
      self.idogramInCartesian()
      data = OrderedDict([('xs',[]),('ys',[]),('fill_color',[]), ('Int',[]), ('Name',[]) ])
      data1 = OrderedDict([('xs',[]),('ys',[]),('fill_color',[]), ('Int',[]), ('Name',[]) ])

      for idx, (x,y) in self.idoCoords.items():
         data1['xs'].append(x)
         data1['ys'].append(y)
         data1['fill_color'].append(self.colors[self.data.columns[idx]])
         data1['Int'].append(1)
         data1['Name'].append(self.data.columns[idx])


      rendsMain = figure.patches(xs="xs",ys="ys",fill_color="fill_color",fill_alpha = 0.7,
                     source = ColumnDataSource(data1),
                     line_color = "darkgrey", line_width = 0.6,
                     hover_fill_color="fill_color",
                     hover_fill_alpha=0.8,
                     )

      n = 0
      for name, lineCoords in self.choords.items():
        n+=1
        for coods in lineCoords:

            x,y = self.connectChoordsInCartesian(coods)
            data['xs'].append(x)
            data['ys'].append(y)
            data['fill_color'].append(self.colors[name])
            data['Int'].append(n)
            data['Name'].append(name)



      self.CDS['main'] = ColumnDataSource(data)

      rends = figure.patches(xs="xs",ys="ys",fill_color="fill_color",fill_alpha = 0.7,
                     source = self.CDS['main'],
                     line_color = "darkgrey", line_width = 0.75,
                     hover_fill_color="fill_color",
                     hover_fill_alpha=0.8,
                     hover_line_color = "black",
                     nonselection_fill_color = "grey",
                    )

      figure.add_tools(TapTool(renderers =  [rends]))
      figure.x_range.start = - self.lim
      figure.x_range.end = self.lim
      figure.y_range.start = - self.lim
      figure.y_range.end = self.lim
      # add text
      r = 1.05
      for name, coords in self.sectors.items():
        theta = np.mean(coords)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        if theta < np.pi/2:
          a = "left"
        if theta > np.pi/2 and theta < 1.5 * np.pi:
          a = "right"
        else:
          a = "left"

        figure.text(x=[x],y=[y],
                  text=[name],
                  text_align = a,
                  text_font_size = "8pt",
                  text_color = "darkgrey" )


  def replot(self, ax = None):
  	""
  	ax.set_xlim(-1.3,1.3)
  	ax.set_ylim(-1.3,1.3)
  	ax.set_aspect('equal')
  	from matplotlib.patches import Polygon
  	self.idogramInCartesian()
  	data1 = OrderedDict([('xs',[]),('ys',[]),('fill_color',[]), ('Int',[]), ('Name',[]) ])
  	for idx, (x,y) in self.idoCoords.items():
  		data1['xs'].append(x)
  		data1['ys'].append(y)
  		df = pd.DataFrame(columns = ["x","y"])
  		df["x"] = x
  		df["y"] = y 
  		xy = df.values

  		ax.add_artist(Polygon(xy, 
  					facecolor = self.colors[self.data.columns[idx]],
  					edgecolor = "darkgrey"))
      
  	data = OrderedDict([('xs',[]),('ys',[]),('fill_color',[]), ('Int',[]), ('Name',[]) ])
  	n = 0
  	for name, lineCoords in self.choords.items():
  		n+=1
  		for coods in lineCoords:
  			x,y = self.connectChoordsInCartesian(coods)
  			data['xs'].append(x)
  			df = pd.DataFrame(columns = ["x","y"])
  			df["x"] = x
  			df["y"] = y
  			xy = df.values
  			ax.add_artist(Polygon(xy, facecolor = self.colors[name], alpha  = 0.65, edgecolor = "darkgrey"))
      
      
        
        

            
            
           # data['ys'].append(y)
           # data['fill_color'].append(self.colors[name])
           # data['Int'].append(n)
           # data['Name'].append(name)	
	
	
    #data = OrderedDict([('xs',[]),('ys',[]),('fill_color',[]), ('Int',[]), ('Name',[]) ])
    
	
	
         
         
         ##data1['fill_color'].append(self.colors[self.data.columns[idx]])
         #data1['Int'].append(1)
         #data1['Name'].append(self.data.columns[idx])
         
		 
		
	
     # rendsMain = figure.patches(xs="xs",ys="ys",fill_color="fill_color",fill_alpha = 0.7,
      #               source = ColumnDataSource(data1),
       #              line_color = "darkgrey", line_width = 0.6,
        #             hover_fill_color="fill_color",
         ##            hover_fill_alpha=0.8,
           #          )
	
	




  def listCoords(self,xList,yList, addMidPoint = True):
    "Assumes that total list length is 2 and adds moints in middle (to do: just measure length)"

    x1,y1 = self.transformToCartesian(0, 0.0*np.pi, transformToListOfPoints = False)
    p = [
        [xList[0],yList[0]],
        [x1,y1],
        [xList[1],yList[1]]
        ]


    return p

  def bezierCurve(self, points, nTimes = 100):
    """
    Source :: Github: https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    Given a set of control points, return the
    bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ]) #nPoints-1

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)


    return xvals, yvals


  def transformToCartesian(self,r,theta, transformToListOfPoints = True):
    "transforms polar to cartesian coordinates"
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if transformToListOfPoints:
      return self.listCoords(x,y)
    else:
      return x, y

  def toPolar(self,x,y):
    "transforms cartesian coords into polar"

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    n = []
    for t in theta:
      if t > 0:
        n.append(t)
      else:
        n.append(2*np.pi - np.abs(t))

    return np.array(n), r



fig = plt.figure()
ax = fig.add_subplot(111)#, projection='polar')



#output_file("chordDiagram.html")

#hover = HoverTool(
#    tooltips = [("Int", "$index"),("n", "@Name")], mode = "mouse")
#hover.point_policy = "follow_mouse"
#f1 = figure(tools = [hover],plot_width=600, plot_height=600,background_fill_color = "#efefef")
#f1.axis.visible = False
#f1.grid.visible = False
df = pd.DataFrame(np.array([#[16,3,28,0,18,2],
                            #[18, 0, 12, 5],
                           # [10, 40, 17, 27],
                           # [19, 0, 35, 11],
                           [5, 80,65],
                           [40, 0,90],
                           [20, 7,0]]
                           ),
          index = ['Complexe I def.', 'Complexe II def.', 'AA starvation'],
          columns = ['Complexe I def.', 'Complexe II def.', 'AA starvation'])
print(df)
choordCoordinates(df).replot(ax)

plt.show()
#show(column(f1))
#plt.show()
