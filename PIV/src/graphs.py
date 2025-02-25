# src/graphs.py

#%% Import Packages

## Calculation packages
import numpy as np

## Graph
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

#%% Functions

def display_PIV_figure(
        image: np.ndarray, 
        x_quiver: np.ndarray,
        y_quiver: np.ndarray,
        u_quiver: np.ndarray,
        v_quiver: np.ndarray,
        display: bool = False,
        display_quiver: bool = True,
        colormap: str = 'gray',
        save: bool = False,
        *args, **kwargs
        ) -> plt.figure:
    
    # Extract kwargs
    save_folder = kwargs.get('save_folder', 'results')
    file_name = kwargs.get('file_name', 'figure')

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(adjust_contrast(image), cmap=colormap)
    if display_quiver:
        plt.quiver(x_quiver, y_quiver, u_quiver, v_quiver, color='red')

    if display:
        plt.show()
    
    if save:
        plt.savefig(result_folder + file_name)

    return fig

def plot_flow_features_map(
        flow_norm: np.ndarray, 
        flow_divergence: np.ndarray, 
        flow_vorticity: np.ndarray,
        result_folder: str = 'results',
        save: bool = True, 
        display: bool = True
        ) -> None:

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    
    for index, value in enumerate([flow_norm, flow_divergence, flow_vorticity]):
        if index == 0: # Norm
            im = ax[index].imshow(value, cmap='viridis', vmin=0, vmax = 10)
            cbar = plt.colorbar(im, ax=ax[index], fraction=0.046, pad=0.04, aspect=20)
            cbar.set_label(f'{["Norm", "Divergence", "Vorticity"][index]} (um/s)')
        else: # Divergence and Vorticity
            im = ax[index].imshow(value, cmap='seismic', norm=CenteredNorm())
            cbar = plt.colorbar(im, ax=ax[index], fraction=0.046, pad=0.04, aspect=20)
            cbar.set_label(f'{["Norm", "Divergence", "Vorticity"][index]}')
        ax[index].set_title(f'Median Flow {["Norm", "Divergence", "Vorticity"][index]}')
        ax[index].axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(result_folder + '/mean_flow_features.png')
    
    if display:
        plt.show()

def plot_flow_features_distribution(
        flow_feature: np.ndarray,
        feature_name: str = 'feature',
        result_folder: str = 'results',
        save: bool = True,
        display: bool = True
        ) -> None:

    fig, ax = plt.subplots(2, 1, figsize=(20,20))

    # Distributions
    f_flatten = flow_feature.flatten()
    hist, bin_edges = np.histogram(f_flatten, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax[0].plot(bin_centers, hist, linewidth=1, color='black')
    ax[0].fill_between(bin_centers, hist, color='black', alpha=0.1)
    med = np.median(f_flatten)
    ax[0].plot(med, np.interp(med, bin_centers, hist), 'o', markersize=10, color='black', alpha=1)
    ax[0].set_title(f'{feature_name} distribution (for all movie frames)')

    # Overtime
    num_slices = flow_feature.shape[0]
    colormap = plt.cm.viridis
    for t in range(num_slices):
        slice_data =flow_feature[t, :, :].flatten()
        color = colormap(t / (num_slices - 1))
        hist, bin_edges = np.histogram(slice_data, bins=15, density=True)

        # Delta for graphical purpose
        x = bin_edges[:-1] + (t * 0.5)  # Décalage en x
        y = hist + (t * 0.005)  # Décalage en y

        ax[1].plot(x, y, linewidth=1, alpha=0.6, color=color)
        med = np.median(slice_data)
        ax[1].plot(med + (t * 0.5), np.interp(med, x, y), 'o', color='black', alpha=0.3)
    ax[1].axis('off')

    if save:
        plt.savefig(result_folder + f'/{feature_name}_distributions.png')
    
    if display:
        plt.show()

def plot_movie_flow_map(
        u: np.ndarray,
        v: np.ndarray,
        result_folder: str = 'results',
        save: bool = True,
        display: bool = True
        ) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    
    um = np.median(u, axis=0)
    vm = np.median(v, axis=0)

    im = ax.imshow(np.sqrt(um **2 + vm **2), cmap='viridis', interpolation='bilinear', vmin=0, vmax=10)
    ax.quiver(range(u.shape[1]), range(u.shape[2]), um, vm, color='r')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label('Speed norm (um/s)')
    
    ax.axis('off')
    ax.set_title(f'{result_folder}.split("/")[-1]')
    plt.tight_layout()

    if save:
        plt.savefig(result_folder + '/PIV_results.png')
    
    if display:
        plt.show() 

#%% Main

def main():
    display_PIV_figure()
    plot_flow_features_map()
    plot_flow_features_distribution()
    plot_movie_flow_map()

if __name__ == "__main__":
    main()