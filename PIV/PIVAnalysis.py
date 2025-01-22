#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:57:22 2025

@author: Laure
"""

# Import packages
import os
from pathlib import Path
import pims
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Images and OpenPIV methods
from openpiv import tools, pyprocess, validation, filters, scaling
import imageio
import importlib_resources
from matplotlib.widgets import Slider

# Get datasets directory
folderPath = Path.cwd().parent.parent / 'datasets' / 'piv'
fileList = os.listdir(folderPath)

# Read image
i = 0
imagePath = 'folderPath' + fileList[i]
frames = pims.open(imagePath)
plt.imshow(frames[0], cmap='gray') #show first frame of the film

# Parameters for PIV
# -- Pre-processing
winsize = 32 # pixels, interrogation window size in frame 't'
searchsize = 38  # pixels, search area size in frame 't+1'
overlap = 17 # pixels, 50% overlap
dt = 0.02 # sec, time interval between the two frames
pxtomm = 9 # pixels/millimeter
# -- Post-processing
s2nThreshold = 1.05 # AU, signal-to-noise threshold to remove false values


# Function: Run PIV on 2 successive images
def pivImageToImage(_frame_a, _frame_b, _winsize, _searchsize, _overlap, _dt, _pxtomm, _s2nThreshold):
	"""
	TBD
	"""

	# --- Pre-processing ---
	# Get the x-axis & y-axis speed together with a signal-to-noise ratio indicating how confident we are in the calculated direction
	u0, v0, sig2noise = pyprocess.extended_search_area_piv(
	    _frame_a.astype(np.int32),
	    _frame_b.astype(np.int32),
	    window_size=_winsize,
	    overlap=_overlap,
	    dt=_dt,
	    search_area_size=_searchsize,
	    sig2noise_method='peak2peak',
	)

	# Get the center of each interrogation window
	x, y = pyprocess.get_coordinates(
	    image_size=_frame_a.shape,
	    search_area_size=_searchsize,
	    overlap=_overlap,
	)

	# --- Post-processing ---
	# Create a mask to remove low signal-to-noise values
	invalid_mask = validation.sig2noise_val(
	    sig2noise,
	    threshold=_s2nThreshold,
	)

	# Replace outliers with local mean values of 'u' and 'v'
	u2, v2 = filters.replace_outliers(
	    u0, v0,
	    invalid_mask,
	    method='localmean',
	    max_iter=3,
	    kernel_size=3,
	)

	# Set correct coordinates and units
	x, y, u3, v3 = scaling.uniform(
	    x, y, u2, v2,
	    scaling_factor=_pxtomm,  
	)

	# 0,0 shall be bottom left, positive rotation rate is counterclockwise
	x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

	# Save as image
	# Display result 
	#fig, ax = plt.subplots(figsize=(8,8))
	#tools.display_vector_field(
	#    pathlib.Path('exp1_001.txt'), # @dev this line should be changed following saving method
	#    ax=ax, scaling_factor=pxtomm,
	#    scale=50, # scale defines here the arrow length
	#    width=0.0035, # width is the thickness of the arrow
	#    on_img=True, # overlay on the image
	#    image_name= str(path / 'data'/'test1'/'exp1_001_a.bmp'),
	#);

	return x, y, u3, v3

# Run PIV on the whole video
x, y, u, v = [], [], [], []
for t in range(len(frames)-1):
	x[t], y[t], u[t], v[t] = pivImageToImage(
		frames[t], 
		frame_b[t+1], 
		winsize, 
		searchsize, 
		overlap, 
		dt, 
		pxtomm, 
		s2nThreshold
	)

	# save film
	# save array
piv_stack

# Display with a Slider
# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Space for slider
img_display = ax.imshow(piv_stack[0], cmap='gray')
ax.set_title("PIV Image Viewer")

# Create slider
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(piv_stack) - 1, valinit=0, valstep=1)

# Update function
def update(val):
    frame = int(slider.val)
    img_display.set_data(piv_stack[frame])
    ax.set_title(f"PIV Image {frame+1}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

# Analysis

