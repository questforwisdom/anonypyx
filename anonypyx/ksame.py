import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import anonypyx.microaggregation as microaggregation

class kSame:
    '''
    Implementation of the k-Same family of image anonymization
    algorithms. 

    [1]: E. M. Newton, L. Sweeney, and B. Malin, ‘Preserving 
    privacy by de-identifying face images’, IEEE Transactions 
    on Knowledge and Data Engineering, vol. 17, no. 2, pp. 
    232–243, Feb. 2005, doi: 10.1109/TKDE.2005.32.

    '''
    def __init__(self, input_images, width, height, **kwargs):
        '''
        Creates a new image anonymizer. Checks whether the parameter values
        are valid.

        Parameters
        ----------

            input_images : numpy.array
                Images to anonymize. Must be a 3-D array in format (image, height, width).
                All images must have the same dimensions and must be grayscale.
                Best results are obtained when key features are in the same position
                in every image (e.g. eyes, nose, mouth for faces).
            width : int
                Width of the input images in pixels.
            height : int
                Height of the input images in pixels.

        Keyword Parameters
        ------------------
            k : int
                Parameter k of k-anonymity. (default: 1)
            variant : string
                k-Same algorithm to be used. Can be either "pixel" or "eigen". (default: "eigen")
            clustering_implementation : string
                Microaggregation algorithm to use. Must be either "MDAV-Generic" or 
                "Random Choice". (default: "Random Choice") 

        Raises
        ------
            TypeError 
                When a parameter has the wrong type.
            ValueError 
                When a single parameter value or a combination of parameter values are invalid.
        '''
        k = kwargs.get("k", 1)
        clustering_implementation = kwargs.get("clustering_implementation", "Random Choice")
        variant = kwargs.get("variant", "eigen")

        if type(k) is not int:
            raise TypeError("k must be an integer.")
        if k < 1 or k > len(input_images):
            raise ValueError("k must be between 1 and the number of images.")

        self.k = k

        if type(width) is not int:
            raise TypeError("width must be an integer.")
        if type(height) is not int:
            raise TypeError("height must be an integer.")

        self.original_shape = (height, width)

        if type(input_images) is not np.ndarray:
            raise TypeError("input_images must be a numpy array.")

        if input_images.shape != (len(input_images), height, width):
            raise ValueError("Input images are malformed. Ensure that every image has the specified width and height.")

        self.input_images = input_images

        if type(variant) is not str:
            raise TypeError("variant must be sa string")
        if variant != "pixel" and variant != "eigen":
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant

        if type(clustering_implementation) is not str:
            raise TypeError("clustering_implementation must be sa string")
        if clustering_implementation == 'Random Choice':
            self.clustering_implementation = microaggregation.RandomChoiceAggregation
        elif clustering_implementation == 'MDAV-Generic':
            self.clustering_implementation = microaggregation.MDAVGeneric
        else: 
            raise ValueError(f"unknown clustering_implementation {clustering_implementation}")

    def anonymize(self):
        '''
        Starts the anonymization algorithm with the options specified by this kSame instance.

        Returns
        -------
            List of anonymized records.
        '''
        self.__setup()

        clustering_algorithm = self.clustering_implementation(self.eigenface_df, self.eigenface_df.columns)
        clusters = clustering_algorithm.partition(self.k)

        anonymized = []

        if (self.variant == 'pixel'):
            anonymized = [self.anonymize_kSamePixel(cluster) for cluster in clusters]
        elif (self.variant == 'eigen'):
            anonymized = [self.anonymize_kSameEigen(cluster) for cluster in clusters]
        else:
            raise ValueError(f"Unknown algorithm variant '{self.variant}' selected.")

        mapping = {image_num: cluster_num for cluster_num in range(len(clusters)) for image_num in clusters[cluster_num]}

        return anonymized, mapping
        

    def __setup(self):
        pixels = self.original_shape[0] * self.original_shape[1]
        flattened_images = np.reshape(self.input_images, (self.input_images.shape[0], pixels))

        self.pca = PCA(n_components = min(self.input_images.shape[0], pixels))
        self.pca.fit(flattened_images)
        eigenfaces = self.pca.transform(flattened_images)

        self.image_df = pd.DataFrame(data = flattened_images)
        self.eigenface_df = pd.DataFrame(data = eigenfaces)

    def anonymize_kSamePixel(self, cluster):
        result = self.image_df.loc[cluster].mean()
        return result.to_numpy().reshape(self.original_shape)

    def anonymize_kSameEigen(self, cluster):
        result = self.eigenface_df.loc[cluster].mean()
        result = self.pca.inverse_transform([result.to_numpy()])[0] 
        return result.reshape(self.original_shape)

