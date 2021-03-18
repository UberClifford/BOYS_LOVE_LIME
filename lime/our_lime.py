### IMPORTS ###
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import felzenszwalb, slic, quickshift
import numpy as np 

### IMAGE ### 
class ImageObject():
    
    def __init__(self, original_image):
        """Initialize image object"""
        
        #keep image as numpy array
        if type(original_image) == np.ndarray:
            self.original_image = original_image
        else:
            self.original_image = np.array(original_image)
        
        self.masked_image = None
        self.superpixels = None
        self.shape = np.array(original_image).shape

    def show(self):
        """Display original image"""
        img = self.original_image
        plt.imshow(img)


### EXPLAINER ###
class Explainer():

    def __init__(self, segmentation_method):
        self.segmentation_method = segmentation_method
    
    def segment_image(self, image, **kwargs):
        """
        image: ImageObject
        """
        image.superpixels = self.segmentation_method(image.original_image, **kwargs)
        
        
    def mask_image(self, image, mask_value = None):
        """
        Generate mask for pixels in image.
        image: ImageObject
        mask_value: If mask_value = None, then each masked pixel is the average
                    of the superpixel it belongs to. Else, every pixel is set to
                    mask_value
        """
        
        img = image.original_image #get original image
        masked_img = img #copy original image
        superpixels = image.superpixels #get original superpixels
        superpixel_ids = np.unique(superpixels) #get superpixels identifiers
        
        #set masked image pixels to average of corresponding superpixel
        if mask_value == None:
            for x in superpixel_ids:
                masked_img[superpixels == x] = np.mean(img[superpixels == x], axis=0)
        #set masked image pixels to mask_value
        else:
            masked_img[:] = mask_value
        
        image.masked_image = masked_img 

### SEGMENTATION ###
class SegmentationMethod():

    def __init__(self, method="quickshift", custom_method=None):
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
        if custom_method:
            self.segmentation_method = custom_method
                
    def __call__(self, img, **kwargs):
        """
        Run skimage segmentation algorithm
        """
        return self.segmentation_method(img, **kwargs)