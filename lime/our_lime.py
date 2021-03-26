### IMPORTS ###
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.segmentation import felzenszwalb, slic, quickshift
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from torchvision import transforms
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
        self.shape = self.original_image.shape

    def show(self):
        """Display original image"""
        plt.imshow(self.original_image)
        plt.imshow(img)


### EXPLAINER ###
class Explainer():

    def __init__(self, classifier, segmentation_method, kernel_method, preprocess_function = None, device = None):
        """
        Inputs:
            preprocess_function: Preprocess function that transforms data to be the same as during
                            blackbox classifier training. If no normalization was used, don't 
                            use this option.
        """
        self.classifier = classifier
        self.segmentation_method = segmentation_method
        self.kernel_method = kernel_method
        if preprocess_function is None:
        if self.preprocess_function is None:
            self.preprocess_function = transforms.Compose([transforms.ToTensor()])
        else:
            self.preprocess_function = preprocess_function
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.classifier = classifier.to(self.device)

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

    def sample_superpixels(self, image, num_samples):
        """
        Samples different configurations of turned on superpixels for the image.
        image: ImageObject
        """
        # sample num_samples collections of superpixels
        num_superpixels = np.unique(image.superpixels).size
        superpixel_samples = np.random.randint(2, size=(num_samples, num_superpixels))

        # apply samples to fudged image to generate pertubed images
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

        Outputs:
            blackbox_io: List of tuples. Each tuple -> (sample_image, blackbox_out)
        """
        sample_labels = list()
        self.classifier.eval()
        with torch.no_grad():
            for sample_image in sampled_images:
                sample_image = torch.unsqueeze(self.preprocess_function(sample_image), dim=0).to(self.device)
                out = self.classifier(sample_image)
                softmax_out = F.softmax(out, dim = 1)
                labels = torch.squeeze(softmax_out, dim = 0).detach().cpu().numpy()
                sample_labels.append(labels)
        sample_labels = np.asarray(sample_labels)

        return sample_labels

    def get_distances(self, superpixel_samples):
        """
        Computes the pairwise distance to the original image superpixel configuration and sampled ones.
        superpixel_samples: thee list of samples of superpixel configurations
        """
        # make an array of ones (i.e. all superpixels on)
        no_mask_array = np.ones(superpixel_samples.shape[1]).reshape(1, -1)
        # distances from each sample to original
        distances = pairwise_distances(superpixel_samples, no_mask_array, metric="cosine")
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

    def fit_LLR(self, samples, weights, labels, regressor = None):
        """
        Fits and returns a regression model to the superpixel samples and the classifier outputs.
        """
        if regressor is None:
            model = Ridge()
        else:
            model = regressor
        model.fit(samples, labels, sample_weight=weights)
        return model

    def explain_image(self, image, num_samples, mask_value = None, top_labels = None, regressor = None):
        if image.superpixels is None:
            self.segment_image(image, num_samples)
        if image.masked_image is None: # What if mask_value changes?
            self.mask_image(image, mask_value)

        #original_blackbox_out = map_blaxbox_io((image,))
        #if top_labels is not None:
        #    labels = original_blackbox_out
        #else:
        #    labels = list(range(len(original_blackbox_out[0])))

        superpixel_samples, sampled_images = sample_superpixels(image)
        blackbox_out = map_blaxbox_io(sampled_images)
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
            self.kernel_method = lambda distances, kernel_width: np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        elif isinstance(method, str):
            raise KeyError(f"Unknown kernel algorithm: {method}")
        else:
            self.kernel_method = method

    def __call__(self, distances):
        """
        Run kernel method on distances
        """
        return self.kernel_method(distances, **self.method_args)
