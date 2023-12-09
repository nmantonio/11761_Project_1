from abc import ABC, abstractmethod
import cv2
import numpy as np
from matplotlib import pyplot as plt



class ImagePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, image):
        pass

    def show_images(self, image):
        prepro_img = self.preprocess(image)
        fig, axes = plt.subplots(1, 2, figsize=(20,20))
        axes = axes.flatten()
        axes[0].imshow(image, cmap='gray', vmin = 0, vmax = 255)
        axes[1].imshow(prepro_img, cmap='gray', vmin = 0, vmax = 255)
        return prepro_img


class BackgroundPreprocessor(ImagePreprocessor):
    def __init__(self, background_points, width=1920, height=1080):
        self.background_points = background_points
        self.background_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self.background_mask, [self.background_points], color=255)
        self.background_mask = cv2.bitwise_not(self.background_mask)

    def preprocess(self, image):
        # Create a mask for the background points
        return cv2.bitwise_and(image, image, mask=self.background_mask)


class ImageAveragingPreprocessor(ImagePreprocessor):
    def __init__(self, background_images):
        self.background_images = [cv2.GaussianBlur(i, ksize=(7,7), sigmaX=0) for i in background_images]
        self.average_image = np.mean(self.background_images, axis=0).astype(np.uint8)

    def preprocess(self, image):
        return image - self.average_image # image - self.average_image #  cv2.absdiff(image, self.average_image)


class ThresholdingPreprocessor(ImagePreprocessor):
    def __init__(self, lower_threshold, upper_threshold):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def preprocess(self, image):
        # Create an inRange mask based on threshold values
        inrange_mask = cv2.inRange(image, self.lower_threshold, self.upper_threshold)
        img = cv2.bitwise_and(image, image, mask=inrange_mask)

        # Apply Otsu's thresholding to further clean the image
        _, img = cv2.threshold(img, self.lower_threshold, self.upper_threshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img
    

class PreprocessingPipeline(ImagePreprocessor):
    def __init__(self, processors):
        self.processors = processors

    def preprocess(self, image):
        for processor in self.processors:
            image = processor.preprocess(image)
        return image
    

class AndBitPreprocessingPipeline(ImagePreprocessor):
    def __init__(self, pipeline1, pipeline2):
        self.pipeline1 = pipeline1
        self.pipeline2 = pipeline2

    def preprocess(self, image):
        i1, i2 = image.copy(), image.copy()
        i1 = self.pipeline1.preprocess(i1)
        i2 = self.pipeline2.preprocess(i2)
        img = cv2.bitwise_and(i1, i2)
        return img
