#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:57:22 2025

@author: Laure
"""

#%% Import Packages

# Import General packages
import os
from pathlib import Path
from tqdm import tqdm

# Import Calculation packages
import numpy as np
import matplotlib.pyplot as plt

# Import Images and OpenPIV methods
import imageio.v2 as imageio
from PIL import Image
from openpiv import tools, pyprocess, validation, filters, scaling
from matplotlib.colors import CenteredNorm
# from matplotlib.widgets import Slider

#%% Run PIV

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
dt = 0.25 # sec, time interval between the two frames
pxtomm = 2.75 # pixels/micrometers
# -- Post-processing
s2nThreshold = 1.05 # AU, signal-to-noise threshold to remove false values

# Function: Run PIV on 2 successive images
def pivImageToImage(
        _frame_a, 
        _frame_b, 
        _winsize, 
        _searchsize, 
        _overlap, 
        _dt, 
        _pxtomm, 
        _s2nThreshold):

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


stacked_image = []

# Run PIV on the whole video
T = frames.shape[0]    
x, y, u, v = [], [], [], []
for t in tqdm(range(T-1)):
    _x, _y, _u, _v = pivImageToImage(
        frames[t,:,:], 
        frames[t+1,:,:], 
        winsize,
        searchsize, 
    	overlap, 
    	dt, 
    	pxtomm, 
    	s2nThreshold
        )
    
    x.append(_x)
    y.append(_y)
    u.append(_u)
    v.append(_v)
    
    displayPIVFigure(frames[t,:,:], _x, _y, _u, _v, display=False)
    plt.close()
    
    # ---- save movie ----
    fig = displayPIVFigure(frames[t,:,:], _x, _y, _u, _v, display=True)
    fig.canvas.draw()
    #img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    img_array = np.array(fig.canvas.renderer.buffer_rgba())  # Get RGBA buffer
    img = Image.fromarray(img_array)  # Convert NumPy array to Image
    stacked_image.append(img)
    plt.close(fig)
    
stacked_image[0].save('multistack.tiff', save_all=True, append_images=stacked_image[1:]) 

x = np.array(x)
y = np.array(y)
u = np.array(u)
v = np.array(v)

#%% Analysis of flow

# norm
def computeNorm(u, v): # compute norm of speed vector
    norm = np.sqrt(u**2 + v**2)
    return norm

# gradients of flow
def computeDivergenceMap(x, y, u, v): # computes divergence ie. if speed goes inward or outward of the zone
    du_dx = np.gradient(u, x, axis=1)  # ∂u/∂x
    dv_dy = np.gradient(v, y, axis=0)  # ∂v/∂y
    div = du_dx + dv_dy
    return div

def computeVorticityMap(x, y, u, v): # compute vorticity ie. if speed tends to rotate clockwise/counter-clockwise around zone
    dv_dx = np.gradient(v, x, axis=1)  # ∂u/∂x
    du_dy = np.gradient(u, y, axis=0)  # ∂v/∂y
    w = dv_dx - du_dy
    return w

flow_norm, flow_divergence, flow_vorticity = [], [], []
for t in tqdm(range(T-1)):
    norm = computeNorm(u[t,:,:], v[t,:,:])
    div = computeDivergenceMap(x[t,0,:], y[t,:,0], u[t,:,:], v[t,:,:])
    w = computeVorticityMap(x[t,0,:], y[t,:,0], u[t,:,:], v[t,:,:])
    
    flow_norm.append(norm)
    flow_divergence.append(div)
    flow_vorticity.append(w)

flowNorm = np.array(flow_norm)
flowDivergence = np.array(flow_divergence)
flowVorticity = np.array(flow_vorticity)

# Median values on 2D map over the course of the film
fig, ax = plt.subplots(1, 3, figsize=(15,5))
meanFlowNorm = np.median(flowNorm, axis=0)
im0 = ax[0].imshow(meanFlowNorm, cmap='viridis')
ax[0].set_title('Median Flow Norm')
plt.colorbar(im0, ax=ax[0])

meanFlowDivergence = np.median(flowDivergence, axis=0)
im1 = ax[1].imshow(meanFlowDivergence, cmap='bwr', norm=CenteredNorm())
ax[1].set_title('Median Flow Divergence')
plt.colorbar(im1, ax=ax[1])

meanFlowVorticity = np.median(flowVorticity, axis=0)
im2 = ax[2].imshow(meanFlowVorticity, cmap='bwr', norm=CenteredNorm())
ax[2].set_title('Median Flow Vorticity')
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()
