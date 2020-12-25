"""
This script classifies the connected components as sigs or non-sigs and
thus produces an output image of extracted signature
Usage: python classify.py <images_path> <save_path> <model>
"""
import cv2
import joblib
import numpy as np
import os
import sys

from tqdm import tqdm

from .features import describe_image
from .utils import list_images


def prepare(filename, classifier):
    # Perform component analysis
    components = describe_image(filename, preprocess=True)

    im = cv2.imread(filename, 0)
    im = np.zeros(im.shape, np.uint8)

    # Perform component analysis
    for (des, idx) in components:
        # Classify each descriptor of the component (to build consensus)
        rows = des.shape[0]
        predictions = np.zeros(rows)
        for row in range(rows):
            predictions[row] = classifier.predict(des[row].reshape(1, -1))

        # Component marked signature only if >99% sure
        confidence = np.count_nonzero(predictions) / len(predictions)
        if confidence < 0.5:
            im[idx] = 255

    return im


def classify(images_path, save_path, model):
    files = list_images(images_path)
    print("Classifying", len(files), "images.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clf = joblib.load(model)

    for fn in tqdm(files):
        mask = prepare(fn, clf)

        outfile = os.path.split(fn)[1]
        outfile = os.path.splitext(outfile)[0] + ".png"

        path = os.path.split(model)[1]
        path = os.path.splitext(path)[0] + "/"
        path = os.path.join(save_path, path)

        outfile = os.path.join(path, outfile)
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

        cv2.imwrite(outfile, mask)
