from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def calculate_f1_score(precision, recall):
    """ Calculate the F1 score based on precision and recall. """
    if precision + recall == 0:
        return 0  # To avoid division by zero
    return 2 * (precision * recall) / (precision + recall)


class Evaluation(ABC):
    @abstractmethod
    def evaluate(self, image, ground_truth, predictions):
        pass

    def show_result(self, image, ground_truth, predictions):
        results = self.evaluate(image, ground_truth, predictions)
        matrix = confusion_matrix([1] * len(ground_truth) + [0] * len(predictions),
                                  [1 if result else 0 for result in results])
        sns.heatmap(matrix, annot=True, fmt='g')
        plt.show()


class NearestPointEuclideanEvaluation(Evaluation):
    def __init__(self, threshold):
        self.threshold = threshold

    def evaluate(self, image, ground_truth, predictions):
        tp, fp, tn, fn = 0, 0, 0, len(ground_truth)
        used_ground_truth = set()  # To keep track of matched ground truth points

        for pred in predictions:
            # Find the nearest ground truth point that hasn't been used
            nearest_dist, nearest_gt = None, None
            for gt in ground_truth:
                if gt not in used_ground_truth:
                    dist = np.linalg.norm(np.array(pred) - np.array(gt))
                    if nearest_dist is None or dist < nearest_dist:
                        nearest_dist = dist
                        nearest_gt = gt

            # Check if nearest ground truth is within the threshold
            if nearest_dist is not None and nearest_dist <= self.threshold:
                tp += 1
                fn -= 1
                used_ground_truth.add(nearest_gt)  # Mark this ground truth as used
            else:
                fp += 1

        return tp, fp, tn, fn


class PeopleCount:
    def __init__(self, preprocessing_pipeline, detector, evaluator):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.detector = detector
        self.evaluator = evaluator

    def run(self, dataframe, show_prediction=False):
        detection_results = []
        evaluation_results = []

        for image_name, group in dataframe.groupby('image_name'):
            image_color = cv2.imread(image_name)
            image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
            preprocessed_image = self.preprocessing_pipeline.preprocess(image, image_color)

            detections = self.detector.detect(preprocessed_image)
            ground_truth = list(zip(group['x'], group['y']))
            evaluation = self.evaluator.evaluate(preprocessed_image, ground_truth, detections)

            detection_results.append({
                'image_name': image_name,
                'detections': detections
            })
            evaluation_results.append({
                'image_name': image_name,
                'evaluation': evaluation
            })

            if show_prediction:
                self.show_predictions(cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB), ground_truth, detections)

        return pd.DataFrame(detection_results), pd.DataFrame(evaluation_results)

    def show_predictions(self, image, ground_truth, detections):
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        prepro_img = self.preprocessing_pipeline.preprocess(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY),
                                                            cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR))
        axes[0].imshow(prepro_img, cmap='gray')
        for x, y in ground_truth:
            cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # Green for ground truth

        for x, y in detections:
            cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Blue for detections
        axes[1].imshow(image, cmap='gray')
        axes[0].set_title(f"Mask after preprocessing.")
        axes[1].set_title(f"Detections (Blue) {len(detections)} vs Ground Truth (Green) {len(ground_truth)}")


def calculate_precision(tp, fp):
    """ Calculate precision based on true positives (TP) and false positives (FP). """
    if tp + fp == 0:
        return 0  # To avoid division by zero
    return tp / (tp + fp)


def calculate_recall(tp, fn):
    """ Calculate recall based on true positives (TP) and false negatives (FN). """
    if tp + fn == 0:
        return 0  # To avoid division by zero
    return tp / (tp + fn)


def display_confusion_matrices(evaluation_results):
    total_confusion_matrix = np.zeros((2, 2), dtype=int)  # For the total confusion matrix
    all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
    for index, row in evaluation_results.iterrows():
        tp, fp, tn, fn = row['evaluation']
        all_tp += tp
        all_fp += fp
        all_tn += tn
        all_tn += fn
        precision = calculate_precision(tp=tp, fp=fp)
        recall = calculate_recall(tp=tp, fn=fn)
        f1_score = calculate_f1_score(precision, recall)
        confusion_matrix = np.array([[tp, fp], [fn, tn]])

        # Display confusion matrix for each image
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', yticklabels=['Positive', 'Negative'],
                    xticklabels=['Positive', 'Negative'])
        plt.title(f"Confusion Matrix for {row['image_name']}, Precision {precision:.5f}, Recall {recall:.5f}, F1-Score {f1_score:.5f}")
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted')
        plt.show()

        # Add to total confusion matrix
        total_confusion_matrix += confusion_matrix

    # Display total confusion matrix
    precision = calculate_precision(tp=all_tp, fp=all_fp)
    recall = calculate_recall(tp=all_tp, fn=all_tn)
    f1_score = calculate_f1_score(precision, recall)
    plt.figure(figsize=(5, 4))
    sns.heatmap(total_confusion_matrix, annot=True, fmt='g', cmap='Blues', yticklabels=['Positive', 'Negative'],
                xticklabels=['Positive', 'Negative'])
    plt.title(f"Total Confusion Matrix, Precision {precision:.5f}, Recall {recall:.5f}, F1-Score {f1_score:.5f}")
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.show()

def calculate_mse(detection_df, annotation_df):
    # Create a DataFrame to store the squared error per image
    img_names = []
    squared_errors = []

    # Iterate through each image in the detection DataFrame
    for image_name in detection_df['image_name'].unique():
        # Get the number of detections for the current image
        detections = len(detection_df[detection_df['image_name'] == image_name]['detections'].iloc[0])

        # Get the number of annotations for the current image
        annotations = len(annotation_df[annotation_df['image_name'] == image_name])

        # Calculate the squared error for the current image
        squared_error = (detections - annotations) ** 2
        img_names.append(image_name)
        squared_errors.append(squared_error)

    # Calculate the mean squared error across all images
    # Append the mean squared error as a separate row
    img_names.append('Mean Squared Error')
    squared_errors.append(np.mean(squared_errors))
    mse_df = pd.DataFrame(data={'Image Name': img_names,
                                'Squared Error': squared_errors})
    return mse_df

