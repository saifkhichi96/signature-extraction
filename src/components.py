from __future__ import print_function
from __future__ import print_function

import argparse
import os

import cv2
import numpy as np


def get_image_files(path):
    """
    Get list of all images in given path (including sub-directories)
    :param path:
    :return:
    """
    images = []
    for f in os.listdir(path):
        fn = os.path.join(path, f)
        if os.path.isdir(fn):
            images += get_image_files(fn)
        else:
            if not f.endswith(".jpeg") and \
                    not f.endswith(".jpg") and \
                    not f.endswith(".png") and \
                    not f.endswith(".tif"):
                continue

            images.append(fn)
    return images


def extract_components(im_binary, connectivity=4):
    # Extract connected components
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(im_binary, connectivity, cv2.CV_32S)

    # Analyse extracted components
    components = dict()
    for label in range(count):
        # Crop out the component
        x, y, w, h = stats[label, 0], stats[label, 1], stats[label, 2], stats[label, 3]
        component = im_binary[y: y + h, x: x + w]

        # The indexes of pixels belonging to char are stored in image
        idx = np.where(labels == label)

        # Extract features from the component
        features = cv2.xfeatures2d.SURF_create()  # SURF features
        features.setHessianThreshold(400)  # with Hessian threshold at 400
        keypoints, descriptors = features.detectAndCompute(component, None)

        components[label] = (descriptors, idx)

    return components


def save_components(components, save_dir, fn):
    """

    :type fn: str
    :type save_dir: str
    :type components: dict
    """
    for label, (descriptor, idx) in components.items():
        if descriptor is not None:
            outfile = os.path.split(fn)[1]
            outfile = os.path.splitext(outfile)[0] + str(label) + ".npy"
            outfile = os.path.join(save_dir, outfile)
            np.save(outfile, descriptor)


if __name__ == '__main__':
    # Command-line arguments parser
    about = "This script extracts SURF features from all images in a given \
            directory and saves them to another specified directory."
    parser = argparse.ArgumentParser(description=about)
    parser.add_argument("images_path", help="path of input images directory")
    parser.add_argument("save_path", help="directory path where individual \
                        components should be saved")
    args = parser.parse_args()

    # Get input and output locations
    images_path = args.images_path
    if not images_path.endswith:
        images_path += "/"

    save_path = args.save_path
    if not save_path.endswith:
        save_path += "/"

    # Create output directory if it does not exist
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    images = get_image_files(images_path)
    print("Found", len(images), "images.")

    # Extract and save all components individually
    for filename in images:
        try:
            # Read image for component extraction
            im = cv2.imread(filename, 0)

            # Thresholding to get binary image
            ret3, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Perform component analysis
            extracted_components = extract_components(thresh, connectivity=8)
            save_components(extracted_components, save_path, filename)
        except:
            print(filename, "is invalid image. Skipped")
