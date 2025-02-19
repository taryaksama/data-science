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

def display_PIV_figure(
        image: np.ndarray, 
        x_quiver: np.ndarray,
        y_quiver: np.ndarray,
        u_quiver: np.ndarray,
        v_quiver: np.ndarray,
        display: bool = False
        ) -> plt.figure:
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(adjust_contrast(image), cmap='gray')
    plt.quiver(x_quiver, y_quiver, u_quiver, v_quiver, color='red')

    if display:
        plt.show()
    
    return fig

def save_image():
    ...

def main() -> None:
    adjust_contrast()
    display_PIV_figure()

if __name__ == '__main__':
    main()