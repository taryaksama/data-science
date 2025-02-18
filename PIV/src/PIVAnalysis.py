#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:57:22 2025

@author: Laure
"""

#%% Import Packages

# Import General packages
from tqdm import tqdm

# Import Calculation packages
import numpy as np
import matplotlib.pyplot as plt

# Import Images and OpenPIV methods
from PIL import Image
from openpiv import tools, pyprocess, validation, filters, scaling
#from matplotlib.colors import CenteredNorm

# Import custom packages
from utils import *

#%% PIV computation

def piv_image2image(
        frame_a: np.ndarray, 
        frame_b: np.ndarray, 
        windsize: int, 
        searchsize: int, 
        overlap: int, 
        dt: float, 
        px2um: float, 
        s2n_threshold: float
        ) -> tuple:

    """
    Run PIV on 2 successive images
    """

	# ---- Pre-processing ----
    # Remove background
    background = (frame_a + frame_b) / 2
    frame_a_rm_background = frame_a - background
    frame_b_rm_background = frame_b - background

	# Get the x-axis & y-axis speed together with a signal-to-noise ratio indicating how confident we are in the calculated direction
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
	    frame_a_rm_background.astype(np.int32),
	    frame_b_rm_background.astype(np.int32),
	    window_size=windsize,
	    overlap=overlap,
	    dt=dt,
	    search_area_size=searchsize,
	    sig2noise_method='peak2peak',
	)

	# Get the center of each interrogation window
    x, y = pyprocess.get_coordinates(
	    image_size=frame_a.shape,
	    search_area_size=searchsize,
	    overlap=overlap,
	)

	# ---- Post-processing ----
	# Create a mask to remove low signal-to-noise values
    invalid_mask = validation.sig2noise_val(
	    u0, v0,
        s2n=sig2noise,
	    threshold=s2n_threshold,
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
	    scaling_factor=px2um,  
	)

	# 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

    return x, y, u3, v3

def piv_movie(
        movie: np.ndarray,
        windsize: int, 
        searchsize: int, 
        overlap: int, 
        dt: float, 
        px2um: float, 
        s2n_threshold: float,
        start: int = 0,
        end: int = movie.shape[0],
        step : int = 1,
        save:bool = False
        ) -> tuple:

    """
    Run PIV on a whole movie
    """

    # ---- PIV computation over the movie ----
    x, y, u, v = [], [], [], []
    for t in tqdm(range(start, end-1, step)):
        _x, _y, _u, _v = piv_image2image(
            movie[t,:,:], 
            movie[t+1,:,:], 
            windsize,
            searchsize, 
            overlap, 
            dt, 
            px2um, 
            s2n_threshold
            )

        x.append(_x)
        y.append(_y)
        u.append(_u)
        v.append(_v)
    
    # Modify type of variables
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)

    # ---- Save PIV results in a multistack TIFF ----
    if save:
        stacked_image = []
        with plt.figure() as fig:
            fig = display_PIV_figure(frames[t,:,:], _x, _y, _u, _v, display=True)
            fig.canvas.draw()
            img_array = np.array(fig.canvas.renderer.buffer_rgba())  # Get RGBA buffer
            img = Image.fromarray(img_array)  # Convert NumPy array to Image
            stacked_image.append(img)
        
        stacked_image[0].save('multistack.tiff', save_all=True, append_images=stacked_image[1:]) 

    return x, y, u, v

#%% Analysis of flow

def compute_norm_map(
        u: np.ndarray,
        v: np.ndarray
        ) -> np.ndarray: 
    """
    compute norm of speed vector
    """

    norm = np.sqrt(u**2 + v**2)
    return norm

def compute_divergence_map(
        x: np.ndarray, 
        y: np.ndarray, 
        u: np.ndarray, 
        v: np.ndarray
        ) -> np.ndarray: 
    """
    computes divergence ie. if speed goes inward or outward of the zone
    """

    du_dx = np.gradient(u, x, axis=1)  # ∂u/∂x
    dv_dy = np.gradient(v, y, axis=0)  # ∂v/∂y
    div = du_dx + dv_dy
    return div

def compute_vorticity_map(
        x: np.ndarray, 
        y: np.ndarray, 
        u: np.ndarray, 
        v: np.ndarray
        ) -> np.ndarray: 
    
    """
    compute vorticity ie. if speed tends to rotate clockwise/counter-clockwise around zone
    """
    
    dv_dx = np.gradient(v, x, axis=1)  # ∂u/∂x
    du_dy = np.gradient(u, y, axis=0)  # ∂v/∂y
    w = dv_dx - du_dy
    return w

def movie_flow_features(
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        start:int = 0,
        end:int = movie.shape[0]
        ) -> tuple:

    """
    Get flow features over the course of the movie
    """

    flow_norm, flow_divergence, flow_vorticity = [], [], []
    for t in tqdm(range(start, end-1)):
        norm = compute_norm_map(u[t,:,:], v[t,:,:])
        div = compute_divergence_map(x[t,0,:], y[t,:,0], u[t,:,:], v[t,:,:])
        w = compute_vorticity_map(x[t,0,:], y[t,:,0], u[t,:,:], v[t,:,:])
        
        flow_norm.append(norm)
        flow_divergence.append(div)
        flow_vorticity.append(w)

    flow_norm = np.array(flow_norm)
    flow_divergence = np.array(flow_divergence)
    flow_vorticity = np.array(flow_vorticity)

    return flow_norm, flow_divergence, flow_vorticity

#%% Main
def main() -> None:
    piv_image2image()
    piv_movie()
    compute_norm_map()
    compute_divergence_map()
    compute_vorticity_map()
    movie_flow_features()

if __name__ == '__main__':
    main()