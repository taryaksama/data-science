# src/utils.py

#%% Import Packages

# Import Calculation packages
import numpy as np
import matplotlib.pyplot as plt

#%%
def select_folder(folder_initial_path:Path = Path.cwd()) -> str:
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(
        initialdir=folder_initial_path,
        title="Select folder")
    return folder_path

def select_file(folder_initial_path:Path = Path.cwd()) -> str:
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        initialdir=folder_initial_path,
        title="Select file",
        filetypes=(("Image files", "*.png *.jpg *.tif *.tiff *.bmp *gif *webp"), ("All files", "*.*"))
    )
    return file_path

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

def main() -> None:
    select_folder()
    select_file()
    adjust_contrast()
    remove_background()

if __name__ == '__main__':
    main()
# %%
