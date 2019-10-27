"""
This script classifies the connected components as sigs or non-sigs and
thus produces an output image of extracted signature
Usage: python classify.py <images_path> <save_path>
"""
from __future__ import print_function

import os
import sys

import cv2
import numpy as np
from sklearn.externals import joblib

import preprocess
from components import extract_components, get_image_files


def __annotate(image, components, classifier):
    image = np.copy(image)

    # Perform component analysis
    for label, (des, idx) in components.items():
        if des is not None:
            # Classify each descriptor of the component (to build consensus)
            rows = des.shape[0]
            predictions = np.zeros(rows)
            for row in range(rows):
                predictions[row] = classifier.predict(des[row].reshape(1, -1))

            # Component marked signature only if >99% sure
            votes_all = len(predictions)
            votes_yes = np.count_nonzero(predictions)
            confidence = 100.0 * votes_yes / votes_all
            if confidence < 50:
                image[idx] = 0
        else:
            image[idx] = 0

    return image


def prepare(filename, classifier):
    im = cv2.imread(filename, 0)

    # todo: crop bottom right of image where signature lies, according to our prior knowledge
    w, h = im.shape
    # im = im[w / 2:w, int(0.60 * h):h]
    # w, h = im.shape

    # Preprocess the image
    im_binary = preprocess.otsu(im)
    im_processed = preprocess.remove_lines(im_binary)
    im_processed_cp = cv2.cvtColor(im_processed, cv2.COLOR_GRAY2BGR)

    # Perform component analysis
    components = extract_components(im_processed, connectivity=8)
    im_annotated = __annotate(im_processed, components, classifier)
    im_annotated_cp = cv2.cvtColor(im_annotated, cv2.COLOR_GRAY2BGR)

    # Calculate mask
    bg_idx = (im_processed_cp == 0)
    fg_idx = (im_annotated_cp != 0)

    # Overlay mask on image
    im_masked = cv2.imread(filename)

    red_mask = np.zeros(im_masked.shape, np.uint8)
    red_mask[:] = (0, 0, 255)

    # im_masked[bg_idx] = 255
    im_masked[fg_idx] = red_mask[fg_idx]

    return cv2.bitwise_not(im_annotated), cv2.cvtColor(im_masked, cv2.COLOR_BGR2RGB)


def segment(image):
    # Approximate bounding box around signature
    points = np.argwhere(image == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
    x, y, w, h = x - 10, y - 10, w + 20, h + 20  # add padding

    # Crop out the signature
    return image[y:y + h, x:x + w]


if __name__ == '__main__':
    if len(sys.argv) not in [4]:
        print('Usage: python classify.py <images_path> <save_path> <model>')
        exit(1)

    images_path = sys.argv[1]
    files = get_image_files(images_path)
    print("Classifying", len(files), "images.")

    save_path = sys.argv[2]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clf = joblib.load(sys.argv[3])

    failed = 0
    i = 0
    for fn in files:
        i += 1
        print("\rProcessing " + str(i) + "/" + str(len(files)) + " ... ", end="")
        sys.stdout.flush()

        im_processed, mask = prepare(fn, clf)
        im_segmented = segment(im_processed)

        # write output image to output folder specified in commandline arguments
        outfile = os.path.split(fn)[1]
        outfile = os.path.splitext(outfile)[0] + ".png"

        path = os.path.split(sys.argv[3])[1]
        path = os.path.splitext(path)[0] + "/"
        path = os.path.join(save_path, path)

        outfile = os.path.join(path, outfile)
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

        h, w = im_segmented.shape
        if w == 0 or h == 0:
            failed += 1
        else:
            cv2.imwrite(outfile, im_segmented)

        print("(" + str(round(100.0 * i / len(files), 2)) + "% done)", end="")

    print("\nSignatures found in %d of %d documents." % (i - failed, i))
