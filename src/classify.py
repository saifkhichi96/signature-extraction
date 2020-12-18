"""
This script classifies the connected components as sigs or non-sigs and
thus produces an output image of extracted signature
Usage: python classify.py <images_path> <save_path> <model>
"""
import os
import sys

import cv2
import joblib
import numpy as np
import preprocess

from features import describe_image
from tqdm import tqdm
from utils import list_images


def __annotate(image, components, classifier):
    image = np.copy(image)

    # Perform component analysis
    for (des, idx) in components:
        if des is not None:
            # Classify each descriptor of the component (to build consensus)
            rows = des.shape[0]
            predictions = np.zeros(rows)
            for row in range(rows):
                predictions[row] = classifier.predict(des[row].reshape(1, -1))

            # Component marked signature only if >99% sure
            confidence = np.count_nonzero(predictions) / len(predictions)
            if confidence < 0.5:
                image[idx] = 0

    return image


def prepare(filename, classifier):
    im = cv2.imread(filename, 0)

    # todo: crop bottom right of image where signature lies, according to our prior knowledge
    # w, h = im.shape
    # im = im[w / 2:w, int(0.60 * h):h]
    # w, h = im.shape

    im_copy = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    # Perform component analysis
    components = describe_image(filename, preprocess=True)
    im_annotated = __annotate(im, components, classifier)
    im_annotated_copy = cv2.cvtColor(im_annotated, cv2.COLOR_GRAY2BGR)

    # Calculate mask
    bg_idx = (im_copy != 0)
    fg_idx = (im_annotated_copy == 0)

    # Overlay mask on image
    im_masked = cv2.imread(filename)

    red_mask = np.zeros(im_masked.shape, np.uint8)
    red_mask[:] = 255

    im_masked[bg_idx] = 0
    im_masked[fg_idx] = red_mask[fg_idx]

    return im, im_masked


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
    files = list_images(images_path)
    print("Classifying", len(files), "images.")

    save_path = sys.argv[2]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    clf = joblib.load(sys.argv[3])

    failed = 0
    i = 0
    for fn in tqdm(files):
        i += 1

        try:
            im_processed, mask = prepare(fn, clf)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            # im_processed = segment(im_processed)

            # write output image to output folder specified in commandline arguments
            outfile = os.path.split(fn)[1]
            outfile = os.path.splitext(outfile)[0] + ".png"

            path = os.path.split(sys.argv[3])[1]
            path = os.path.splitext(path)[0] + "/"
            path = os.path.join(save_path, path)

            outfile = os.path.join(path, outfile)
            if not os.path.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))

            h, w, _ = mask.shape
            if w == 0 or h == 0:
                failed += 1
            else:
                cv2.imwrite(outfile, mask)
        except Exception as ex:
            print(ex)
            failed += 1

    print("\nSignatures found in %d of %d documents." % (i - failed, i))
