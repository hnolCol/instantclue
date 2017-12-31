import configparser
import os 
def load_config_file(path='D:\MS_PYStasi\settings\default_settings.ini'):

     config = configparser.ConfigParser()
     config.read(path)

     return config._sections 
