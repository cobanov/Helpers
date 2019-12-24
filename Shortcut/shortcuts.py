"""
Author: Mert Cobanoglu // MSI-GA
Date:   3.10.2019

This script can delete unwanted shorcuts 
and change the wallpaper to black screen.

"""

import os
from pathlib import Path
import ctypes



# Change Wallpaper 

SPI_SETDESKWALLPAPER = 20 
ctypes.windll.user32.SystemParametersInfoA(SPI_SETDESKWALLPAPER, 0, "" , 0)


# Delete Unwanted Shortcuts

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
files = os.listdir(desktop)

delete = []

for i in delete:
    try:
        os.remove(desktop + "\\" + i)
    except FileNotFoundError:
        continue
    
