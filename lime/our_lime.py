### IMPORTS ###
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.segmentation import felzenszwalb, slic, quickshift
from sklearn.metrics import pairwise_distances
import numpy as np 
import torch
import torch.nn.functional as F

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

    def __init__(self, classifier, segmentation_method, kernel_method, num_samples=1000, preprocess_function = None):
        self.classifier = classifier
        self.segmentation_method = segmentation_method
        self.kernel_method = kernel_method
        self.num_samples = num_samples
        if preprocess_function is None:
            self.preprocess_function = transforms.Compose([transforms.ToTensor()])

    def segment_image(self, image):
        """
        image: ImageObject
        """
        image.superpixels = self.segmentation_method(image.original_image)        

    def mask_image(self, image, mask_value = None):
        """
        Generate mask for pixels in image.
        image: ImageObject
        mask_value: If mask_value = None, then each masked pixel is the average
                    of the superpixel it belongs to. Else, every pixel is set to
                    mask_value
        """

        img = image.original_image #get original image
        masked_img = img.copy() #copy original image
        superpixels = image.superpixels #get original superpixels
        superpixel_ids = np.unique(superpixels) #get superpixels identifiers

        #set masked image pixels to average of corresponding superpixel
        if mask_value is None:
            for x in superpixel_ids:
                masked_img[superpixels == x] = np.mean(img[superpixels == x], axis=0)
        #set masked image pixels to mask_value
        else:
            masked_img[:] = mask_value

        image.masked_image = masked_img 

    def sample_superpixels(self, image):
        """
        Samples different configurations of turned on superpixels for the image.
        image: ImageObject
        """
        # sample num_samples collections of superpixels
        num_superpixels = np.unique(image.superpixels).size
        superpixel_samples = np.random.randint(2, size=(self.num_samples, num_superpixels))

        # apply samlpes to fudged image to generate pertubed images
        sampled_images = list()
        for sample in superpixel_samples:
            sample_masked_image = image.original_image.copy()
            turned_on_superpixels = np.where(sample == 1)[0]
            mask = np.zeros(image.superpixels.shape).astype(bool)
            for superpixel in turned_on_superpixels:  # turn on the sampled pixels
                mask[image.superpixels == superpixel] = True
            sample_masked_image[mask] = image.masked_image[mask]
            sampled_images.append(sample_masked_image)
        return superpixel_samples, sampled_images

    def map_blaxbox_io(self, sampled_images):    
        """
        Inputs:
            sampled_images: Image samples resulting from different superpixel combinations.
                            List of numpy arrays (rows, col, 3). 
            preprocess_function: Preprocess function that transforms data to be the same as during
                                 blackbox classifier training. If no normalization was used, don't 
                                 use this option.
        Outputs:
            blackbox_io: List of tuples. Each tuple -> (sample_image, blackbox_out)
        """
        blackbox_out = list()
        self.classifier.eval()
        for sample_image in sampled_images:
            sample_image = torch.unsqueeze(preprocess_function(sample_image), dim=0)
            out = self.classifier(sample_image)
            softmax_out = F.softmax(out, dim = 1)
            labels = torch.squeeze(softmax_out.detach(), dim = 0).numpy()
            blackbox_out.append(labels)
        blackbox_io = list(zip(sampled_images, blackbox_out))

        return blackbox_io

    def get_distances(self, superpixel_samples):
        """
        Computes the pairwise distance to the original image superpixel configuration and sampled ones.
        superpixel_samples: thee list of samples of superpixel configurations
        """
        # make an array of ones (i.e. all superpixels on)
        no_mask_array = np.ones(superpixel_samples.shape[1]).reshape(1, -1)
        # distances from each sample to original
        distances = pairwise_distances(superpixel_samples, no_mask_array, metric="euclidean")
        return distances.flatten()

    def weigh_samples(self, distances):
        """
        Inputs:
            distances: 1D numpy array. Sample distances to original data point.

        Outputs:
            sample_weights:  1D numpy array. Sample distances weighed by kernel method.
        """
        sample_weights = self.kernel_method(distances)
        return sample_weights

    def select_features(self):
        pass

    def fit_LLR(blackbox_io, weights, labels, regressor = None):
        pass

    def explain_image(self, image, mask_value = None, top_labels = None, regressor = None):
        if image.superpixels is None:
            self.segment_image(image)
        if image.masked_image is None: # What if mask_value changes?
            self.mask_image(image, mask_value)
        superpixel_samples, sampled_images = sample_superpixels(self, image)
        blackbox_io = map_blaxbox_io(self, sampled_images)
        distances = get_distances(superpixel_samples)
        sample_weights = weigh_samples(distances)
        # select_features()
        # fit_LLR()
        # create explanation


### SEGMENTATION ###
class SegmentationMethod():

    def __init__(self, method="quickshift", **method_args):
        """
        Set image segmentation method as a predefined algorithm or custom function
        """
        self.method = method
        self.method_args = method_args

        if self.method == "quickshift":
            self.segmentation_method = quickshift
        elif self.method == "felzenszwalb":
            self.segmentation_method = felzenszwalb
        elif self.method == "slic":
            self.segmentation_method = slic
        elif isinstance(method, str):
            raise KeyError(f"Unknown segmentation algorithm: {method}")
        else:
            self.segmentation_method = method
                
    def __call__(self, img):
        """
        Run segmentation method on image
        """
        return self.segmentation_method(img, **self.method_args)


### SIMILARITY KERNEL ###
class KernelMethod():

    def __init__(self, method="exponential", **method_args):
        """
        Set similarity kernel method as a predefined algorithm or custom function
        """
        self.method = method
        self.method_args = method_args

        if self.method == "exponential":
            self.kernel_method = lambda distances, kernel_width: np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        elif isinstance(method, str):
            raise KeyError(f"Unknown kernel algorithm: {method}")
        else:
            self.kernel_method = method

    def __call__(self, distances):
        """
        Run kernel method on distances
        """
        return self.kernel_method(distances, **self.method_args)
