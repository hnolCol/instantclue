#!/usr/bin/env python
"""
    Module for plot annotation methods.

    Created Jan 2014
    Copyright (C) Damien Farrell

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

from __future__ import absolute_import, division, print_function
'''try:
    from tkinter import *
    from tkinter.ttk import *
except:
    from Tkinter import *
    from ttk import *'''
import types
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict

colormaps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))

def addTextBox(fig, kwds=None):
    """Add a rectangle"""

    #from . import handlers
    import matplotlib.patches as patches
    from matplotlib.text import OffsetFrom
    self.applyOptions()
    if kwds == None:
        kwds = self.kwds
    fig = self.parent.fig
    #ax = self.parent.ax
    ax = fig.get_axes()[0]
    canvas = self.parent.canvas
    text = kwds['text'].strip('\n')
    fc = kwds['facecolor']
    ec = kwds['linecolor']
    pad=kwds['pad']
    bstyle = kwds['boxstyle']
    style = "%s,pad=%s" %(bstyle,pad)
    fontsize = kwds['fontsize']
    font = mpl.font_manager.FontProperties(family=kwds['font'],
                        weight=kwds['fontweight'])
    bbox_args = dict(boxstyle=bstyle, fc=fc, ec=ec, lw=1, alpha=0.9)
    arrowprops = dict(arrowstyle="->", connectionstyle="arc3")

    an = ax.annotate(text, xy=(.5, .5), xycoords='axes fraction',
               ha=kwds['align'], va="center",
               size=fontsize,
               fontproperties=font,
               bbox=bbox_args)
    an.draggable()
    canvas.show()
    if text not in self.textboxes:
        self.textboxes[text] = kwds
        self.objects[text] = an
    print (self.textboxes)
    return

    def saveCoords(self, an):
        """Save the coords of current annotations for redrawing"""

        x = bbox.x0
        y = bbox.y0
        return