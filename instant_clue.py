"""
	""MAIN FILE - THIS FILE STARTS THE PROGRAM""
    Instant Clue - Interactive Data Visuali5zation and Analysis.
    Copyright (C) Hendrik Nolte

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""


import sys
import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg

import webbrowser
import urllib.request as urllibReq

import start_page
import analyze_data

import multiprocessing

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from modules.utils import *

class instantClueApp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        if platform == 'WINDOWS':
            tk.Tk.iconbitmap(self, default=os.path.join(path_file,'icons','logo_ic.ico'))

        tk.Tk.wm_title(self, "Interactive Data Analysis - CECAD Cologne")
        self.protocol("WM_DELETE_WINDOW", self.close_up)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = self.create_menu(container)
        tk.Tk.config(self, menu=menubar)

        self.frames = {}
        if True:
            for F in (start_page.StartPage,analyze_data.analyze_data):

                frame = F(container, self)

                self.frames[F] = frame

                frame.grid(row=0, column=0, sticky="nsew")
                if F == start_page.StartPage:
                        frame.grid_columnconfigure(0,weight=1)
                        frame.grid_columnconfigure(3,weight=1)
                        frame.grid_rowconfigure(1,weight=1)
                        frame.grid_rowconfigure(6, weight=1)
                if F == analyze_data.analyze_data :
                        frame.grid_columnconfigure(3, weight=1, minsize=200)
                        frame.grid_rowconfigure(5, weight=1, minsize=345)
                        frame.grid_rowconfigure(11, weight=1, minsize=70)
                        frame.grid_columnconfigure(0,weight=3,minsize=400)


        self.show_frame(start_page.StartPage)
        #self.check_for_new_version()




    def create_menu(self, container):
        '''
        Creates tkinter menu.
        '''
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Exit", command=self.close_up)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Tutorial", command=lambda:webbrowser.open(webisteUrlTutorial))
        helpmenu.add_command(label="You Tube Videos", command=lambda:webbrowser.open(websiteUrlVideos))
        menubar.add_cascade(label="Help", menu=helpmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Orig. Publication", command = lambda: webbrowser.open(paperUrl))
        menubar.add_cascade(label="Read more", menu=helpmenu)

        return menubar

    def check_for_new_version(self):
    	'''
    	Extracts the version from the html page (release_information.html) that holds
    	information about recently updates. It simply checks takes the text from the
    	website and compares it to the current software version.
    	'''
    	update = False
    	try:
    		releasePage = urllibReq.urlopen('http://www.instantclue.uni-koeln.de/release_information.html')
    		text = str(releasePage.read())
    		currentVersion = start_page.__VERSION__.split('.')
    		textVersion = text.split('Version')[1].split('<')[0]
    		textSplitVersion = textVersion.split('.')
    	except:
    		return

    	if len(currentVersion) != len(textSplitVersion):
    		return
    	
    	for n,num in enumerate(textSplitVersion):
    		if float(num) > float(currentVersion[n]):
    			if n != 0:
    				if float(currentVersion[n-1]) >= float(textSplitVersion[n-1]):
    					pass
    				else:
    					continue
    			update = True
    			break
    	if update:
    		quest = tk.messagebox.askquestion('New version available ..',
    								  'There is a new version available {}.'.format(textVersion) +
    								  ' Would you like to download? Upon download extract the zip folder.'+
    								  ' You may delete this version. Thank you for using Instant Clue.',
    								  parent = self)

    		if quest == 'yes':
    			webbrowser.open('http://www.instantclue.uni-koeln.de/download/InstantClue_{}.zip'.format(platform.lower()))

    def show_frame(self, cont):
        '''
        Raise frame.
        '''
        frame = self.frames[cont]
        frame.tkraise()

    def close_up(self, quest = None):
        '''
        Close the application.
    	'''
        if quest is None:
        	quest =tk.messagebox.askquestion('Close..',
        						'Closing InstantClue.\nPlease confirm.')
        if quest == 'yes':
             for frame in self.frames.values():
                 frame.destroy()

             instantClueApp.destroy(self)
             app.quit()
        else:
            return


if __name__ == "__main__":

     multiprocessing.freeze_support()
     app = instantClueApp()
     if platform == 'LINUX':
     	#enable cascade extension automatically on linux systems
     	app.bind_class("Menu", "<<MenuSelect>>", activate_cascade)
     	app.style  = ttk.Style()
     	app.style.theme_use("clam")

     screen_width, screen_height = app.winfo_screenwidth(), app.winfo_screenheight()
     w = 1690
     h = 1080
     appGeom = evaluate_screen(screen_width,screen_height,w,h)
     app.geometry(appGeom)
     # method after is not needed here but works much better
     # with focusing afterwards frames have been created
     app.after(40,app.check_for_new_version)
     app.mainloop()
     sys.exit(1)
     os._exit(1)
