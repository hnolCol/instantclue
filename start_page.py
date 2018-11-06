
import sys
import os
import tkinter as tk
import webbrowser 

import analyze_data


from tkinter import ttk
from modules import images
from modules.utils import *


__VERSION__ = '0.5.2' #Date 12.09.2018


class StartPage(tk.Frame):

     def __init__(self,parent, controller):
          tk.Frame.__init__(self,parent, background="white")
          
          self.get_images()
          self.build_widgets(controller)
          
          
     def get_images(self):
          '''
          Get images from image module.
          '''

          self.data_analysis_icon,self.webisteLogo, self.videoTutorialLogo, \
          self.pdfTutorialLogo, self.sourceCodeLogo, self.LOGO = images.get_start_page_images()         						  
    
     def build_widgets(self, controller):
          '''
          Defines and grid widgets controller.
          '''
    	
          self.label_LOGO = tk.Label(self, bg="white" ,image=self.LOGO)

          self.lab_version = tk.Label(self,
                              text = 'Version '+__VERSION__,
                              bg="white",
                              font = NORM_FONT,
                              fg ="#4C626F")
          
          self.labelWelcome = tk.Label(self, text="Welcome To Instant Clue \n\nInteractive "+
          				   "Scientific Visualization and Analysis\n\n\n",
                           font = LARGE_FONT, fg="#4C626F", justify=tk.LEFT, bg="white")
							
                           
          self.websiteButton = create_button(self,image=self.webisteLogo, 
          								command = lambda: webbrowser.open(websiteUrl))
          self.videoTutorialButton = create_button(self,image=self.videoTutorialLogo,
          								command = lambda: webbrowser.open(websiteUrlVideos))
          self.pdfTutorialButton = create_button(self,image=self.pdfTutorialLogo,
          								command = lambda: webbrowser.open(webisteUrlTutorial))
        
          self.sourceCodeButton = create_button(self,image=self.sourceCodeLogo , 
          								command = lambda: webbrowser.open(gitHubUrl))
          self.analyze_data_but = create_button(self, image= self.data_analysis_icon,
          								command=lambda: controller.show_frame(analyze_data.analyze_data))
    
          
          self.label_cite = tk.Label(self, text = "If you found usage of Instant Clue helpful"+
          						"; please cite :\nNolte, H., MacVicar D. T., Frederik T., and Krüger, M."+
          						" - Instant Clue: A Software Suite for Interactive Data Visualization and Analysis",
                                bg="white",
                                fg="#4C626F",
                                justify=tk.LEFT,
                                font = NORM_FONT)
                                
          make_label_button_like(self.label_cite)
          self.label_cite.bind('<Button-1>', lambda event: webbrowser.open(paperUrl)) 
                                
        

  
          ## grid widgets
          self.labelWelcome.grid(row=0, column=0, padx=10, pady=20, columnspan=1, sticky=tk.NW )
          self.analyze_data_but.grid(row = 2,column=0, padx=(50,25),sticky=tk.E+tk.N, pady=(50,11),rowspan=5)#, sticky=tk.E) 
          
          self.websiteButton.grid(row=2,column=1,sticky=tk.N,pady=(50,9))
          self.videoTutorialButton.grid(row=3,column=1,sticky=tk.N,pady=9)
          self.pdfTutorialButton.grid(row=4,column=1,sticky=tk.N,pady=9)
          self.sourceCodeButton.grid(row=5,column=1,sticky=tk.N,pady=9)
              
                    
          self.label_LOGO.grid(row=0, column=2, sticky=tk.N + tk.E, padx=0, pady=10, columnspan=2)         
          self.label_cite.grid(row=6,column=0, padx=15, sticky=tk.SW, columnspan=4, pady=20)
          self.lab_version.grid(row=6,column=3, padx=15, sticky=tk.E+tk.S, pady=20)


        
