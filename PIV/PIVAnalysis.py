#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:57:22 2025

@author: Laure
"""

# Import General packages
import os
from pathlib import Path
from tqdm import tqdm

# Import Calculation packages
import numpy as np
import matplotlib.pyplot as plt

# Import Images and OpenPIV methods
import imageio.v2 as imageio
from openpiv import tools, pyprocess, validation, filters, scaling
# from matplotlib.widgets import Slider

# Get datasets directory
folderPath = Path.cwd().parent.parent / 'datasets' / 'piv' / 'background_subtracted_trial' 
fileList = os.listdir(folderPath)

# Read image
i = 0
imagePath = folderPath / fileList[i]
frames = imageio.imread(imagePath)

def adjustContrast(image):
    min_val = np.min(image)
    max_val = np.max(image)
    
    adjusted_image = 255 * (image-min_val) / (max_val-min_val)
    adjusted_image = adjusted_image.astype(np.uint8)
    
    return adjusted_image

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
	    u0, v0,
        s2n=sig2noise,
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
	    scaling_factor=1,  
	)

	# 0,0 shall be bottom left, positive rotation rate is counterclockwise
	x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

	return x, y, u3, v3

def displayPIVFigure(background_image, x_quiver, y_quiver, u_quiver, v_quiver, display=False):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(adjustContrast(background_image), cmap='gray')
    plt.quiver(x_quiver, y_quiver, u_quiver, v_quiver, color='red')

    if display==True:
        plt.show()
    
    return fig

# Run PIV on the whole video
T = 6 #@dev, frames.shape[0]
L = 56 #@dev, to be matched with PIV parameters
x = np.zeros((T,L,L), dtype=np.int64)
y = np.zeros((T,L,L), dtype=np.int64)
u = np.zeros((T,L,L), dtype=np.int64)
v = np.zeros((T,L,L), dtype=np.int64)

for t in tqdm(range(T-1)):
    x[t,:,:], y[t,:,:], u[t,:,:], v[t,:,:] = pivImageToImage(
        frames[t,:,:], 
        frames[t+1,:,:], 
        winsize,
        searchsize, 
    	overlap, 
    	dt, 
    	pxtomm, 
    	s2nThreshold
        )
    background_image = frames[t,:,:]
    displayPIVFigure(background_image, x[t,:,:], y[t,:,:], u[t,:,:], v[t,:,:], display=False)
    plt.close()



# Analysis of flow

# orientation
# theta = atan(v/u)

# gradients of flow
# du_dx = np.gradient(u, x, axis=1)  # ∂u/∂x
# du_dy = np.gradient(u, x, axis=0)  # ∂u/∂y
# dv_dx = np.gradient(v, y, axis=1)  # ∂v/∂x
# dv_dy = np.gradient(v, y, axis=0)  # ∂v/∂y

# divergence
# div = ∂u/∂x + ∂v/∂y

# vorticity
# w = ∂v/∂x - ∂u/∂y

# mean values on 2D map over the course of the film
# np.mean(div, axis=2)
