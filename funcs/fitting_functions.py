# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:42:43 2017

@author: hnolte-101
"""
import numpy as np


def build_equation(mode,sel_,error):
         
             if error is not None:                  
                 error = [round(err,1) for err in error]
             mode = [round(mode_val,1) for mode_val in mode]
             if 'poly_fit' in sel_ or 'linear' in sel_:
                 deg = len(list(mode))
                 base = 'f(x)= ' 
                 for i in range(deg):                    
                         power = deg-1-i
                         
                         if power == 1:
                             attach = str(mode[i])+'x+'
                         elif power == 0:
                             attach = str(mode[i])
                         else:    
                             attach = str(mode[i])+'x^'+str(power)+'+'
                         base = base + attach
                 return base        
           
                 
             elif 'A exp(b*x)' in sel_:
                 build_ex = '{}\pm{}\cdot x'.format(mode[1],error[1])
                 build_am = '{}\pm{}'.format(mode[0],error[0])
                 #build_const =  '{}\pm{}'.format(round(mode[2],1),round(error[2],1))
                 base = 'f(x) = '+build_am+'e^{'+build_ex+'}'
                 return base
             elif 'A (1 - exp(-k * x))' in sel_:
                 
                 build_ex = '-{}\pm{}\cdot x'.format(mode[1],error[1])
                 build_am = '{}\pm{}'.format(mode[0],error[0])
                 build_const =  '{}\pm{}'.format(mode[2],error[2])
                 base = 'f(x) = '+build_am+'\cdot (1-e^{'+build_ex+'}) +'+ build_const
                 return base
             elif 'A exp(b*x) + C exp(d*x)' in sel_:
                        build_A_1 = '{}\pm{}'.format(mode[0],error[0])
                        build_A_2 = '{}\pm{}'.format(mode[2],error[2])
                        build_ex_1 = '{}\pm{}\cdot x'.format(mode[1],error[1])
                        build_ex_2 = '{}\pm{}\cdot x'.format(mode[3],error[3])
                        base = 'f(x) = '+build_A_1+'\cdot e^{'+build_ex_1+'}+'+build_A_2+'\cdot e^{'+build_ex_2+'}'
             elif 'Michaelis Menten (Vmax*x)/(Km+x)' in sel_:
                 top_ = '{}\pm{}\cdot x'.format(mode[0],error[0])
                 bottom_ = '{}\pm{}+ x'.format(mode[1],error[1])
                 base = r'v = \frac{{'+top_+'}}{{'+bottom_+'}}' #'f(x) = '+build_am+'e^{'+build_ex+'}+'+build_const
                 return base
             elif 'A b^(x) + c' in sel_:
                 build_a = '{}\pm{}'.format(mode[0],error[0])
                 build_b = '{}\pm{}'.format(mode[1],error[1])
                 build_c = '{}\pm{}'.format(mode[2],error[2])
                 
                 base = 'f(x)= '+build_a+'\cdot '+build_b+'^{x}'+build_c
                 return base
             
def michaelis_menten(x,Vmax,Km):
    """
    x - Substrate concentration
    Vmax - maximal reaction rate
    Km - Michaelis constant
    """
    a = Vmax * x
    b = Km + x
    return (a) / (b)


def exponential_fit_1term(x,a,b):
    """
    x = any data
    a Amplitude
    b mult
    y = a * exp(b*x)
    """
    return a*np.exp(b*x)

def cosine_fit(x,amplitude,phase,offset):
	"""
	A = Amplitude
	freq = Frequency
	phase = Phase
	offset = Offset y axis
	"""
	return amplitude*np.cos((x*(2*np.pi)/23.6) + phase)+offset
	


def exponential_fit_2term(x,a,b,c,d):
    """
    x = any data
    a Amplitude
    b mult
    y = a * exp(b*x)
    """
    return a*np.exp(b*x) + c*np.exp(d*x)

def exponential_fit_non_e(x,a,b,c):
    """
    """
    return a*b**x+c


def exponential_fit_one_e(x,A,k,y0):
    """
    f(x) = y = A [1-exp(-kx)] + y0
    """
    return A*(1-np.exp(-k*x)) + y0


def gaussian_fit(x,A,mu,sigma):
    """
    Fit gaussian 
    """
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))



 