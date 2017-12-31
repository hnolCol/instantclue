import sys
import os

sys.path.append('.../funcs/')
from funcs import resize_window_det

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import webbrowser
import start_page
import analyze_data

import multiprocessing
#multiprocessing.set_start_method('forkserver')
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

                    
        self.show_frame(start_page.StartPage)
        
    def create_menu(self, container):
		
        
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        
        menubar.add_cascade(label="File", menu=filemenu)
        

        filemenu.add_command(label="Exit", command=self.close_up)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Tutorial", command=lambda:webbrowser.open(webisteUrlTutorial))
        menubar.add_cascade(label="Help", menu=helpmenu)
          
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Orig. Publication", command = lambda: webbrowser.open('http://google.de'))
        menubar.add_cascade(label="Read more", menu=helpmenu)
        
        return menubar
        
        
	
    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()
        
    def close_up(self):
        quest =tk.messagebox.askquestion('Close..','Closing InstantClue.\nPlease confirm.')
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
     screen_width, screen_height = app.winfo_screenwidth(), app.winfo_screenheight()
     w = 1620
     h = 1080
     appGeom = evaluate_screen(screen_width,screen_height,w,h)     
     app.geometry(appGeom)
     app.mainloop() 
     sys.exit(1)
     os._exit(1)
    
