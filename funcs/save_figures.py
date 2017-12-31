import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from tkinter import messagebox



def save_all(figures,save_path_name,curr = 'all'):

     plt.rcParams['pdf.fonttype'] = 42
     matplotlib.rcParams['pdf.fonttype'] = 42
     try:
          pp = PdfPages(save_path_name)

          if curr == 'last':

              pp.savefig(figures[-1])  
              
          else:
               for i in figures:
                    pp.savefig(i)
          pp.close()
     except (OSError, IOError) as e:
          messagebox.showinfo('Error ...',e)
          return 

          
          
