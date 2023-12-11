from abc import ABC, abstractmethod

import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImagePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, image_grayscale=None, image_color=None):
        pass

    def show_images(self, image_grayscale=None, image_color=None):
        if image_grayscale is None and image_color is None:
            raise Exception("NO input images!")
        prepro_img = image_grayscale.copy()
        prepro_img = self.preprocess(prepro_img, image_color.copy())
        fig, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes = axes.flatten()
        axes[0].imshow(image_grayscale, cmap='gray', vmin=0, vmax=255)
        axes[1].imshow(prepro_img, cmap='gray', vmin=0, vmax=255)
        return prepro_img


class BackgroundPreprocessor(ImagePreprocessor):
    def __init__(self, background_points, width=1920, height=1080):
        self.background_points = background_points
        self.background_mask = np.zeros((height, width), dtype=np.uint8)
        for each in self.background_points:
            cv2.fillPoly(self.background_mask, [each], color=255)
        self.background_mask = cv2.bitwise_not(self.background_mask)

    def preprocess(self, image_grayscale=None, image_color=None):
        # Create a mask for the background points
        if image_grayscale is None:
            raise Exception("No input grayscale image!")
        return cv2.bitwise_and(image_grayscale, image_grayscale, mask=self.background_mask)


class ImageAveragingPreprocessor(ImagePreprocessor):
    def __init__(self, background_images):
        self.background_images = [cv2.GaussianBlur(i, ksize=(7, 7), sigmaX=0) for i in background_images]
        self.average_image = np.mean(self.background_images, axis=0).astype(np.uint8)

    def preprocess(self, image_grayscale=None, image_color=None):
        if image_grayscale is None:
            raise Exception("No input grayscale image!")
        return image_grayscale - self.average_image


class ThresholdingPreprocessor(ImagePreprocessor):
    def __init__(self, lower_threshold, upper_threshold):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def preprocess(self, image_grayscale=None, image_color=None):
        if image_grayscale is None:
            raise Exception("No input grayscale image!")
        # Create an inRange mask based on threshold values
        inrange_mask = cv2.inRange(image_grayscale, self.lower_threshold, self.upper_threshold)
        img = cv2.bitwise_and(image_grayscale, image_grayscale, mask=inrange_mask)

        # Apply Otsu's thresholding to further clean the image
        _, img = cv2.threshold(img, self.lower_threshold, self.upper_threshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img


class WaterThresholdingPreprocessor(ImagePreprocessor):
    def __init__(self,
                 lower_blue=np.array([90, 40, 40]),
                 upper_blue=np.array([130, 255, 255])) -> None:
        self.lower_blue = lower_blue
        self.upper_blue = upper_blue

    def preprocess(self, image_grayscale=None, image_color=None):
        if image_grayscale is None and image_color is None:
            raise Exception("No input grayscale and color images!")
        hsv_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
        sea_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)

        removed_sea = image_grayscale.copy()
        removed_sea[sea_mask > 0] = [0]

        return removed_sea


class SandThresholdingPreprocessor(ImagePreprocessor):
    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def preprocess(self, image_grayscale=None, image_color=None):
        if image_grayscale is None:
            raise Exception("No input grayscale image!")
        _, sand_mask = cv2.threshold(image_grayscale, self.threshold, 255, cv2.THRESH_BINARY_INV)

        removed_sand = image_grayscale.copy()
        removed_sand[sand_mask > 0] = [0]
        return removed_sand


class SobelPreprocessor(ImagePreprocessor):
    def __init__(self, gradient_threshold=200) -> None:
        self.gradient_threshold = gradient_threshold

    def preprocess(self, image_grayscale=None, image_color=None):
        if image_grayscale is None:
            raise Exception("No input grayscale image!")
        sobel_x = cv2.Sobel(image_grayscale, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_grayscale, cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        _, binary_gradient = cv2.threshold(gradient_magnitude, self.gradient_threshold, 255, cv2.THRESH_BINARY)
        binary_gradient = 255 - binary_gradient

        result = image_grayscale.copy()
        result[binary_gradient == 0] = [0]
        return result


class CLAHEPreprocessor(ImagePreprocessor):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def preprocess(self, image_grayscale=None, image_color=None):
        return self.clahe.apply(image_grayscale)


class OtsuPreprocessor(ImagePreprocessor):
    def __init__(self) -> None:
        pass

    def preprocess(self, image_grayscale=None, image_color=None):
        if image_grayscale is None:
            raise Exception("No input grayscale image!")
        _, binary_image = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image


class PreprocessingPipeline(ImagePreprocessor):
    def __init__(self, processors) -> None:
        self.processors = processors

    def preprocess(self, image_grayscale=None, image_color=None):
        image = image_grayscale.copy()
        for processor in self.processors:
            image = processor.preprocess(image, image_color)
        return image


class MultipleANDBitProcessingPipeline(ImagePreprocessor):
    def __init__(self, pipelines) -> None:
        self.pipelines = pipelines

    def preprocess(self, image_grayscale=None, image_color=None):
        images = []
        for p in self.pipelines:
            images.append(p.preprocess(image_grayscale.copy(), image_color.copy()))

        base_img = images[0]
        for i in range(1, len(images)):
            base_img = cv2.bitwise_and(base_img, images[i])
        return base_img