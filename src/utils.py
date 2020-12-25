import cv2
import numpy as np
import os


def list_images(path, formats=['jpeg', 'jpg', 'png', 'tif', 'tiff']):
    """Lists all images in a directory (including sub-directories).

    Images in JPG, PNG and TIF format are listed.
    """
    images = []
    for f in os.listdir(path):
        fn = os.path.join(path, f)
        if os.path.isdir(fn):
            images += list_images(fn)
        else:
            ext = f.split('.')[-1]
            for format in formats:
                if ext.lower() == format.lower():
                    images.append(fn)
                    break

    return sorted(images)


def iou(y_pred, y_true):
    intersection = np.count_nonzero(cv2.bitwise_and(y_pred, y_true))
    union = np.count_nonzero(cv2.bitwise_or(y_pred, y_true))
    return intersection / union

def dice(y_pred, y_true):
    a = iou(y_pred, y_true)
    return 2*a / (a+1)

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
