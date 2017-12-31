import sys
import os
import tkinter as tk
import webbrowser 

import analyze_data
from funcs import resize_window_det
from funcs import determine_platform


from tkinter import ttk
from modules import images
from modules.utils import *


__VERSION__ = '0.3.756' #Date 15.12.2017



class StartPage(tk.Frame):
     def __init__(self,parent, controller):
          tk.Frame.__init__(self,parent, background="white")
          self.get_images()
          self.build_widgets(controller)
          
          
          
     def get_images(self):

          self.data_analysis_icon,self.webisteLogo, self.videoTutorialLogo, \
          self.pdfTutorialLogo, self.sourceCodeLogo, self.LOGO = images.get_start_page_images()         						  
    
     def build_widgets(self, controller):
    	
          StartPage.version = __VERSION__
          self.label_LOGO = tk.Label(self, bg="white" ,image=self.LOGO)

          self.lab_version = tk.Label(self,
                              text = 'Version '+__VERSION__,
                              bg="white",
                              font = NORM_FONT,
                              fg ="#4C626F")
          
          self.labelWelcome = tk.Label(self, text="Welcome To Instant Clue \n\nInteractive Scientific Visualization and Analysis\n\n\n",
                           font=TITLE_FONT,
                           bg="white",
                           fg="#4C626F",
                           justify=tk.LEFT)
                           
          self.websiteButton = create_button(self,image=self.webisteLogo, 
          								command = lambda: webbrowser.open(websiteUrl))
          self.videoTutorialButton = create_button(self,image=self.videoTutorialLogo,
          								command = lambda: webbrowser.open(websiteUrlVideos))
          if platform == 'MAC':
          	
          	self.pdfTutorialButton = create_button(self,image=self.pdfTutorialLogo,
          								command = lambda: os.system("open {}".format(tutorialPath)))
          else:
          	
          	self.pdfTutorialButton = create_button(self,image=self.pdfTutorialLogo, 
          								command = lambda: os.system("start {}".format(tutorialPath)))
          self.sourceCodeButton = create_button(self,image=self.sourceCodeLogo , 
          								command = lambda: webbrowser.open(gitHubUrl))
          self.analyze_data_but = create_button(self, image= self.data_analysis_icon,
          								command=lambda: controller.show_frame(analyze_data.analyze_data))
    
          
          self.label_cite = tk.Label(self, text = "If you found usage of Instant Clue helpful; please cite :\nNolte, H., MacVicar, D. T. and Krüger, M. INSTANt Clue - INteractive ScienTific ANalysis: A Software Suite For Scientific Data Visualization and Analysis by Drag & Drop",
                                bg="white",
                                fg="#4C626F",
                                justify=tk.LEFT,
                                font = NORM_FONT)

  
          ## grid widgets
          self.labelWelcome.grid(row=0, column=0, padx=10, pady=20, columnspan=1, sticky=tk.NW )
          self.analyze_data_but.grid(row = 2,column=0, padx=(50,25),sticky=tk.E+tk.N, pady=(50,11),rowspan=5)#, sticky=tk.E) 
          
          self.websiteButton.grid(row=2,column=1,sticky=tk.N,pady=(50,11))
          self.videoTutorialButton.grid(row=3,column=1,sticky=tk.N,pady=11)
          self.pdfTutorialButton.grid(row=4,column=1,sticky=tk.N,pady=11)
          self.sourceCodeButton.grid(row=5,column=1,sticky=tk.N,pady=12)
              
                    
          self.label_LOGO.grid(row=0, column=2, sticky=tk.N + tk.E, padx=0, pady=10, columnspan=2)         
          self.label_cite.grid(row=6,column=0, padx=15, sticky=tk.SW, columnspan=4, pady=20)
          self.lab_version.grid(row=6,column=3, padx=15, sticky=tk.E+tk.S, pady=20)


        
