### IMPORTS ###
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

### IMAGE ### 

class ImageObject():
    
    def __init__(self, original_image_path):
        """Initialize image object"""
        self.original_image_path = original_image_path
        self.fudge_image = None
        self.superpixels = None
            
    def change_attribute(self, new_fugde_value = None, new_superpixel_value = None):
        """Change attribute when fugde images and superpixels are computed in Explainer Class"""
        if new_fugde_value != None:
            self.fudge_image =  new_fugde_value
        if new_superpixel_value != None:
            self.superpixels = new_superpixel_value

    def show(self):
        """Display original image"""
        img = self.original_image_path
        with open(img, 'rb') as f:
            with Image.open(f) as img:
                img.convert('RGB')
                plt.imshow(img)

### EXPLAINER ###




### SEGMENTATION ###
