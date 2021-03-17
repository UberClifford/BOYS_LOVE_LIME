### IMPORTS ###
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import felzenszwalb, slic, quickshift

### IMAGE ### 

class ImageObject():
    
    def __init__(self, original_image):
        """Initialize image object"""
        self.original_image = original_image
        self.fudge_image = None
        self.superpixels = None
                    
    def change_attribute(self, new_fugde_value = None, new_superpixel_value = None):
        """Change attribute when fugde images and superpixels are computed in Explainer Class"""
        if new_fugde_value is not None:
            self.fudge_image =  new_fugde_value
        if new_superpixel_value is not None:
            self.superpixels = new_superpixel_value

    def show(self):
        """Display original image"""
        img = self.original_image
        plt.imshow(img)

### EXPLAINER ###


### SEGMENTATION ###
class SegmentationMethod():

    def __init__(self, method):
        """
        Set segmentation method to skimage.segmentation method
        """
        self.method = method        
        if self.method == "quickshift":
            self.segmentation_method = quickshift
        elif self.method == "felzenszwalb":
            self.segmentation_method = felzenszwalb
        elif self.method == "slic":
            self.segmentation_method = slic
                
    def segment(self, img, **kwargs):
        """
        Run skimage segmentation algorithm
        """
        segments = self.segmentation_method(img, **kwargs)
        return segments
    