# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:53:54 2017
Main Work: 
@author: https://github.com/Phlya/adjustText/blob/master/adjustText/adjustText.py
"""
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
from operator import itemgetter

def adjust_text(texts,x_data,y_data, renderer = None, ax = None,force_text=0.3, add_objects=None, force_points=0.3, force_objects=0.3,expand_align=(0.8, 0.8),autoalign='xy',
                     va='bottom', ha='left',only_move={}, text_from_text=True, text_from_points=True,expand_points=(1.3, 1.3),expand_text=(1.3, 1.3),expand_objects=(1.3, 1.3),
                                                        save_steps=False, lim=30,precision=2.5, add_bboxes  = []):
         def get_text_position(text, ax=None):
            ax = ax or plt.gca()
            x, y = text.get_position()
            return (ax.xaxis.convert_units(x),
                    ax.yaxis.convert_units(y))

         def get_bboxes(objs, r, expand=(1.0, 1.0), ax=None):
            if ax is None:
                ax = plt.gca()
            return [i.get_window_extent(r).expanded(*expand).transformed(ax.\
                                                  transData.inverted()) for i in objs]
        
         def get_midpoint(bbox):
            cx = (bbox.x0+bbox.x1)/2
            cy = (bbox.y0+bbox.y1)/2
            return cx, cy
        
         def get_points_inside_bbox(x, y, bbox):
            x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            
            x_in = np.logical_and(x>x1, x<x2)
            y_in = np.logical_and(y>y1, y<y2)
            return np.where(x_in & y_in)[0]
        
         def get_renderer(fig):
            try:
                return fig.canvas.get_renderer()
            except AttributeError:
                return fig.canvas.renderer
        
         def overlap_bbox_and_point(bbox, xp, yp):
            cx, cy = get_midpoint(bbox)
        
            dir_x = np.sign(cx-xp)
            dir_y = np.sign(cy-yp)
        
            if dir_x == -1:
                dx = xp - bbox.xmax
            elif dir_x == 1:
                dx = xp - bbox.xmin
            else:
                dx = 0
        
            if dir_y == -1:
                dy = yp - bbox.ymax
            elif dir_y == 1:
                dy = yp - bbox.ymin
            else:
                dy = 0
            return dx, dy
        
         def move_texts(texts, delta_x, delta_y, bboxes=None, renderer=None, ax=None):
            if ax is None:
                ax = plt.gca()
            if bboxes is None:
                if renderer is None:
                    r = get_renderer(ax.get_figure())
                else:
                    r = renderer
                bboxes = get_bboxes(texts, r, (1, 1), ax=ax)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            for i, (text, dx, dy) in enumerate(zip(texts, delta_x, delta_y)):
                bbox = bboxes[i]
                x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
                if x1 + dx < xmin:
                    dx = 0
                if x2 + dx > xmax:
                    dx = 0
                if y1 + dy < ymin:
                    dy = 0
                if y2 + dy > ymax:
                    dy = 0
        
                x, y = get_text_position(text, ax=ax)
                newx = x + dx
                newy = y + dy
                text.set_position((newx, newy))
            return texts     
         def repel_text(texts, renderer=None, ax=None, expand=(1.2, 1.2),
               only_use_max_min=False, move=False):
                """
                Repel texts from each other while expanding their bounding boxes by expand
                (x, y), e.g. (1.2, 1.2) would multiply width and height by 1.2.
                Requires a renderer to get the actual sizes of the text, and to that end
                either one needs to be directly provided, or the axes have to be specified,
                and the renderer is then got from the axes object.
                """
                if ax is None:
                    ax = plt.gca()
                if renderer is None:
                    r = get_renderer(ax.get_figure())
                else:
                    r = renderer
                bboxes = get_bboxes(texts, r, expand, ax=ax)
                xmins = [bbox.xmin for bbox in bboxes]
                xmaxs = [bbox.xmax for bbox in bboxes]
                ymaxs = [bbox.ymax for bbox in bboxes]
                ymins = [bbox.ymin for bbox in bboxes]
               
                overlaps_x = np.zeros((len(bboxes), len(bboxes)))
                overlaps_y = np.zeros_like(overlaps_x)
                overlap_directions_x = np.zeros_like(overlaps_x)
                overlap_directions_y = np.zeros_like(overlaps_y)
                for i, bbox1 in enumerate(bboxes):
                    overlaps = get_points_inside_bbox(xmins*2+xmaxs*2, (ymins+ymaxs)*2,
                                                         bbox1) % len(bboxes)
                    overlaps = np.unique(overlaps)
                    for j in overlaps:
                        bbox2 = bboxes[j]
                        x, y = bbox1.intersection(bbox1, bbox2).size
                        overlaps_x[i, j] = x
                        overlaps_y[i, j] = y
                        direction = np.sign(bbox1.extents - bbox2.extents)[:2]
                        #print(direction)
                        overlap_directions_x[i, j] = direction[0]
                        overlap_directions_y[i, j] = direction[1]
            
                move_x = overlaps_x*overlap_directions_x
                move_y = overlaps_y*overlap_directions_y
            
                delta_x = move_x.sum(axis=1)
                delta_y = move_y.sum(axis=1)
            
                q = np.sum(np.abs(delta_x) + np.abs(delta_y))
                if move:
                    move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
                return delta_x, delta_y, q
            
         def repel_text_from_bboxes(add_bboxes, texts, renderer=None, ax=None,
                                       expand=(1.2, 1.2), only_use_max_min=False,
                                       move=False):
                """
                Repel texts from other objects' bboxes while expanding their (texts')
                bounding boxes by expand (x, y), e.g. (1.2, 1.2) would multiply width and
                height by 1.2.
                Requires a renderer to get the actual sizes of the text, and to that end
                either one needs to be directly provided, or the axes have to be specified,
                and the renderer is then got from the axes object.
                """
                if ax is None:
                    ax = plt.gca()
                if renderer is None:
                    r = get_renderer(ax.get_figure())
                else:
                    r = renderer
            
                bboxes = get_bboxes(texts, r, expand, ax=ax)
            
                overlaps_x = np.zeros((len(bboxes), len(add_bboxes)))
                overlaps_y = np.zeros_like(overlaps_x)
                overlap_directions_x = np.zeros_like(overlaps_x)
                overlap_directions_y = np.zeros_like(overlaps_y)
            
                for i, bbox1 in enumerate(bboxes):
                    for j, bbox2 in enumerate(add_bboxes):
                        try:
                            x, y = bbox1.intersection(bbox1, bbox2).size
                                                     
                            direction = np.sign(bbox1.extents - bbox2.extents)[:2]
                            overlaps_x[i, j] = x
                            overlaps_y[i, j] = y
                            overlap_directions_x[i, j] = direction[0]
                            overlap_directions_y[i, j] = direction[1]
                        except AttributeError:
                            pass
            
                move_x = overlaps_x*overlap_directions_x
                move_y = overlaps_y*overlap_directions_y
            
                delta_x = move_x.sum(axis=1)
                delta_y = move_y.sum(axis=1)
            
                q = np.sum(np.abs(delta_x) + np.abs(delta_y))
                if move:
                    move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
                return delta_x, delta_y, q
            
         def repel_text_from_points(x, y, texts, renderer=None, ax=None,
                                       expand=(1.2, 1.2), move=False):
                """
                Repel texts from all points specified by x and y while expanding their
                (texts'!) bounding boxes by expandby  (x, y), e.g. (1.2, 1.2)
                would multiply both width and height by 1.2.
                Requires a renderer to get the actual sizes of the text, and to that end
                either one needs to be directly provided, or the axes have to be specified,
                and the renderer is then got from the axes object.
                """
                assert len(x) == len(y)
                if ax is None:
                    ax = plt.gca()
                if renderer is None:
                    r = get_renderer(ax.get_figure())
                else:
                    r = renderer
                bboxes = get_bboxes(texts, r, expand, ax=ax)
            
                move_x = np.zeros((len(bboxes), len(x)))
                move_y = np.zeros((len(bboxes), len(x)))
                for i, bbox in enumerate(bboxes):
                    xy_in = get_points_inside_bbox(x, y, bbox)
                    #print(xy_in)
                    for j in xy_in:
                        xp, yp = x[j], y[j]
                        dx, dy = overlap_bbox_and_point(bbox, xp, yp)
                       
                        move_x[i, j] = dx
                        move_y[i, j] = dy
            
                delta_x = move_x.sum(axis=1)
                delta_y = move_y.sum(axis=1)
                q = np.sum(np.abs(delta_x) + np.abs(delta_y))
                if move:
                    move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
                return delta_x, delta_y, q
            
         def repel_text_from_axes(texts, ax=None, bboxes=None, renderer=None,
                                     expand=None):
                if ax is None:
                    ax = plt.gca()
                if renderer is None:
                    r = get_renderer(ax.get_figure())
                else:
                    r = renderer
                if expand is None:
                    expand = (1, 1)
                if bboxes is None:
                    bboxes = get_bboxes(texts, r, expand=expand, ax=ax)
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
                    dx, dy = 0, 0
                    if x1 < xmin:
                        dx = xmin - x1
                    if x2 > xmax:
                        dx = xmax - x2
                    if y1 < ymin:
                        dy = ymin - y1
                    if y2 > ymax:
                        dy = ymax - y2
                    if dx or dy:
                        x, y = get_text_position(texts[i], ax=ax)
                        newx, newy = x + dx, y + dy
                        texts[i].set_position((newx, newy))
                return texts   
         def optimally_align_text(x, y, texts, expand=(1., 1.), add_bboxes=[],
                         renderer=None, ax=None,
                         direction='xy'):
                """
                For all text objects find alignment that causes the least overlap with
                points and other texts and apply it
                """
                if ax is None:
                    ax = plt.gca()
                if renderer is None:
                    r = get_renderer(ax.get_figure())
                else:
                    r = renderer
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                bboxes = get_bboxes(texts, r, expand, ax=ax)
                if 'x' not in direction:
                    ha = ['']
                else:
                    ha = ['center', 'left', 'right']
                if 'y' not in direction:
                    va = ['']
                else:
                    va = ['bottom', 'top', 'center']
                alignment = list(product(ha, va))
                for i, text in enumerate(texts):

                    counts = []
                    for h, v in alignment:
                        if h:
                            text.set_ha(h)
                        if v:
                            text.set_va(v)
                        bbox = text.get_window_extent(r).expanded(*expand).\
                                                   transformed(ax.transData.inverted())
                        c = len(get_points_inside_bbox(x, y, bbox))
                        intersections = [bbox.intersection(bbox, bbox2) for bbox2 in
                                         bboxes+add_bboxes]
                        intersections = sum([abs(b.width*b.height) if b is not None else 0
                                             for b in intersections])
                        # Check for out-of-axes position
                        bbox = text.get_window_extent(r).transformed(ax.transData.inverted())
                        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
                        if x1 < xmin or x2 > xmax or y1 < ymin or y2 > ymax:
                            axout = 1
                        else:
                            axout = 0
                        counts.append((axout, c, intersections))
                    a, value = min(enumerate(counts), key=itemgetter(1))
                    if 'x' in direction:
                        text.set_ha(alignment[a][0])
                    if 'y' in direction:
                        text.set_va(alignment[a][1])
                    bboxes[i] = text.get_window_extent(r).expanded(*expand).\
                                                   transformed(ax.transData.inverted())
                    return texts
         def float_to_tuple(a):
            try:
                a = float(a)
                return (a, a)
            except TypeError:
                assert len(a)==2
                assert all([bool(i) for i in a])
                return a
         def get_text_position(text, ax=None):
             
                ax = ax or plt.gca()
                x, y = text.get_position()
                return (ax.xaxis.convert_units(x),
                        ax.yaxis.convert_units(y))
                
         if ax == None:
             ax = plt.gca()  
         r = renderer
         orig_xy = [get_text_position(text, ax=ax) for text in texts]
         orig_x = [xy[0] for xy in orig_xy]
         orig_y = [xy[1] for xy in orig_xy] 
         force_objects = float_to_tuple(force_objects)
         force_text = float_to_tuple(force_text)
         force_points = float_to_tuple(force_points)
         x, y = orig_x, orig_y
         if add_objects is None:
            text_from_objects = False
            add_bboxes = []
         else:
             try:
                 add_bboxes = get_bboxes(add_objects, r, ax=ax)
                 text_from_objects = True
                 
             except:
                 add_bboxes = []
                 text_from_objects = False
                 pass
             
             
             
         x1 = x_data
         y1 = y_data
 
         texts = repel_text_from_axes(texts, ax, renderer=r, expand=expand_points)
         history = [np.inf]*5
         for i in range(lim):
             q1, q2 = np.inf, np.inf
             if True:
                 d_x_text, d_y_text, q1 = repel_text(texts, renderer=r, ax=ax,
                                                expand=expand_text)
             else:
                d_x_text, d_y_text, q1 = [0]*len(texts), [0]*len(texts), 0
    
             if True:
                d_x_points, d_y_points, q2 = repel_text_from_points(x = x1, y= y1, texts = texts,
                                                       ax=ax, renderer=r,
                                                       expand=expand_points)
             else:
                d_x_points, d_y_points, q2 = [0]*len(texts), [0]*len(texts), 0
             if text_from_objects:
                d_x_objects, d_y_objects, q3 = repel_text_from_bboxes(add_bboxes,
                                                                      texts,
                                                                 ax=ax, renderer=r,
                                                             expand=expand_objects)
             else:
                d_x_objects, d_y_objects, q3 = [0]*len(texts), [0]*len(texts), 0
             dx = (np.array(d_x_text) * force_text[0] +
                  np.array(d_x_points) * force_points[0] +
                  np.array(d_x_objects) * force_objects[0])
             dy = (np.array(d_y_text) * force_text[1] +
                  np.array(d_y_points) * force_points[1] +
                  np.array(d_y_objects) * force_objects[1])
             q = round(q1+q2+q3, 5)
             if q > precision and q < np.max(history):
                    history.pop(0)
                    history.append(q)
                    move_texts(texts, dx, dy,
                               bboxes = get_bboxes(texts, r, (1, 1)), ax=ax)
         return texts, orig_xy
         
         
         
