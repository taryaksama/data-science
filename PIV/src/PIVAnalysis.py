# src/PivAnalysis.py

#%% Import Packages

# Import General packages
import os
from datetime import datetime
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
    frame_a_rm_background = adjust_contrast(remove_background(frame_a, background, method='divide'))
    frame_b_rm_background = adjust_contrast(remove_background(frame_b, background, method='divide'))

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

    # Get the filtering ratio
    mask = invalid_mask[2]
    filtering_ratio = np.sum(mask) / np.size(mask)

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

    return x, y, u3, v3, filtering_ratio

def piv_movie(
        movie: np.ndarray,
        windsize: int, 
        searchsize: int, 
        overlap: int, 
        dt: float, 
        px2um: float, 
        s2n_threshold: float,
        start: int = 0,
        end: int = None,
        step : int = 1,
        save:bool = False,
        *args, **kwargs
        ) -> tuple:

    """
    Run PIV on a whole movie
    """

    # Extract kwargs
    save_folder = kwargs.get('save_folder', 'results')

    if end is None:
        end = movie.shape[0]

    # ---- PIV computation over the movie ----
    x, y, u, v = [], [], [], []
    stacked_image = []
    for t in tqdm(range(start, end-1, step)):
        _x, _y, _u, _v, _ = piv_image2image(
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

        # ---- Save PIV results in a multistack TIFF ----
        if save:
            fig = display_PIV_figure(movie[t,:,:], _x*px2um, _y*px2um, _u, _v, display=False)
            fig.canvas.draw()
            img_array = np.array(fig.canvas.renderer.buffer_rgba())  # Get RGBA buffer
            img = Image.fromarray(img_array)  # Convert NumPy array to Image
            stacked_image.append(img)
            plt.close()
    
    # Modify type of variables
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)

    if save:        
        stacked_image[0].save(save_folder + '/PIV_movie.tiff', save_all=True, append_images=stacked_image[1:]) 

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
        end:int = None
        ) -> tuple:

    """
    Get flow features over the course of the movie
    """

    if end is None:
        end = u.shape[0]

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

def flow_features_mean_map(
        flow_norm: np.ndarray,
        flow_divergence: np.ndarray,
        flow_vorticity: np.ndarray,
        save: bool = False,
        *args, **kwargs
        ) -> tuple:
    
    # Extract kwargs
    save_folder = kwargs.get('save_folder', 'results')

    # Median values on 2D map over the course of the film
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    # Norm
    mean_flow_norm = np.median(flow_norm, axis=0)
    im0 = ax[0].imshow(mean_flow_norm, cmap='viridis')
    ax[0].set_title('Median Flow Norm')
    plt.colorbar(im0, ax=ax[0])

    # Divergence
    mean_flow_divergence = np.median(flow_divergence, axis=0)
    im1 = ax[1].imshow(mean_flow_divergence, cmap='bwr', norm=CenteredNorm())
    ax[1].set_title('Median Flow Divergence')
    plt.colorbar(im1, ax=ax[1])

    # Vorticity
    mean_flow_vorticity = np.median(flow_vorticity, axis=0)
    im2 = ax[2].imshow(mean_flow_vorticity, cmap='bwr', norm=CenteredNorm())
    ax[2].set_title('Median Flow Vorticity')
    plt.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig(result_folder+ '/mean_flow_features.png')
    
    return mean_flow_norm, mean_flow_divergence, mean_flow_vorticity
        

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