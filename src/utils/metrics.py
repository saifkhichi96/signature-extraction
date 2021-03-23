import cv2
import numpy as np

from .utils import list_images


def iou(y_pred, y_true):
    intersection = np.count_nonzero(cv2.bitwise_and(y_pred, y_true))
    union = np.count_nonzero(cv2.bitwise_or(y_pred, y_true))
    return intersection / union


def dice(y_pred, y_true):
    a = iou(y_pred, y_true)
    return 2 * a / (a + 1)


def jaccard_score(predictions, groundtruth):
    y_preds = list_images(predictions)
    y_trues = list_images(groundtruth)
    if len(y_preds) != len(y_trues):
        raise Exception()

    score = []
    for i, y_pred in enumerate(y_preds):
        y_pred = cv2.imread(y_pred, 0)
        y_true = cv2.imread(y_trues[i], 0)
        score.append(iou(y_pred, y_true))

    return np.mean(score)


def f1_score(predictions, groundtruth):
    y_preds = list_images(predictions)
    y_trues = list_images(groundtruth)
    if len(y_preds) != len(y_trues):
        raise Exception()

    score = []
    for i, y_pred in enumerate(y_preds):
        y_pred = cv2.imread(y_pred, 0)
        y_true = cv2.imread(y_trues[i], 0)
        score.append(dice(y_pred, y_true))

    return np.mean(score)
