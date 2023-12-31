from abc import ABC, abstractmethod

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class GenericDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image):
        """
        Abstract method to detect objects in an image.
        This method should be implemented by subclasses to perform actual detection.

        Parameters:
        image (numpy.ndarray): A grayscale image in which to perform the detection.

        Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates for each detected object.
        """
        pass

    def show_detections(self, image):
        """
        Perform detection on the image and show the image with detected objects marked.

        Parameters:
        image (numpy.ndarray): The image on which to perform and show detections.
        """
        plt.figure(figsize=(16, 12))
        detections = self.detect(image)
        plt.imshow(image, cmap='gray')
        for x, y in detections:
            plt.scatter(x, y, c='green', s=30)
        plt.show()


class SimpleBlobDetector(GenericDetector):
    def __init__(self,
                 params=None):
        super().__init__()
        self.params = cv2.SimpleBlobDetector_Params() if params is None else params
        self.detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, image):
        # Detect blobs.
        keypoints = self.detector.detect(image)

        # Convert keypoints to (x, y) coordinates
        return [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]


class HOGSvmDetector:
    def __init__(self):
        # Initialize HOG descriptor with default people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image):
        # Detect people in the image
        boxes, _ = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
        center_points = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in boxes]

        return center_points


class HaarCascadeDetector(GenericDetector):
    def __init__(self, cascade_path):
        super().__init__()
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        # Detect objects (faces) in the image
        objects = self.cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Extract the top-left corner coordinates of each detected object
        return [(x, y) for (x, y, w, h) in objects]


class ContourDetector(GenericDetector):
    def __init__(self, threshold=200):
        super().__init__()
        self.threshold = threshold

    def detect(self, image):
        # Detect objects (faces) in the image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for contour in contours:

            # Ignore small contours (adjust the threshold as needed)
            if cv2.contourArea(contour) < self.threshold:
                continue

            # Calculate the center of mass (centroid) of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                objects.append((cx, cy))

        return objects


class DBSCANHumanDetector(GenericDetector):
    def __init__(self, eps=30, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def detect(self, image):
        # image = cv2.Canny(image, 100, 255)
        # Extract features (e.g., points of interest)
        features = self.extract_features(image)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(features)

        # Calculate centroids of clusters
        centroids = self.calculate_centroids(features, clustering.labels_)

        return centroids

    def extract_features(self, image):
        # Implement feature extraction here
        # Example: Extract coordinates of non-zero points
        points = np.column_stack(np.where(image > 0))
        points = np.flip(points, axis=1)
        return points

    def calculate_centroids(self, features, labels):
        centroids = []
        for cluster_id in np.unique(labels):
            if cluster_id != -1:  # Ignoring noise points
                # Extract points belonging to the current cluster
                points_in_cluster = features[labels == cluster_id]
                # Calculate the centroid
                centroid = np.mean(points_in_cluster, axis=0).astype(int)
                centroids.append(tuple(centroid))
        return centroids
