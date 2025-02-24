# src/utils.py

#%% Import Packages

# Import Calculation packages
import numpy as np
import matplotlib.pyplot as plt

#%%

def adjust_contrast(image: np.ndarray) -> np.ndarray:
    min_val = np.min(image)
    max_val = np.max(image)
    
    adjusted_image = 255 * (image-min_val) / (max_val-min_val)
    adjusted_image = adjusted_image.astype(np.uint8)
    
    return adjusted_image

def remove_background(
        image: np.ndarray, 
        background: np.ndarray,
        method: str = 'divide'
        ) -> np.ndarray:
    
    if method == 'divide':
        return image / background
    if method == 'subtract':
        return image - background

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

def main() -> None:
    adjust_contrast()
    display_PIV_figure()

if __name__ == '__main__':
    main()
# %%
