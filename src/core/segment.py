import cv2
import joblib
import numpy as np
import os

from tqdm import tqdm

from .features import get_components
from utils import list_images


def extract_signature(img, clf, preprocess=True):
    """Segments signatures in an image.

    This function receives a grayscale image as input, from which all connected
    components are extracted and then classified as either a signature or not.
    A mask image of same size as the input image is generated, with signature
    pixels in white (intensity 255) and background in black (intensity 0).

    Parameters:
        img : the image to extract signature from
        clf : the classifier to use for predicting component class

    Returns:
        the segmentation mask with signature pixels in white
    """
    # Extract SURF features of connected components
    components = get_components(img, preprocess)

    # Classify each component as signature/background
    mask = np.zeros(img.shape, np.uint8)
    for (descriptors, idx) in components:
        # A component may have multiple descriptors. Classify each
        # of them separately.
        n_descriptors = descriptors.shape[0]
        predictions = np.zeros(n_descriptors)
        for i in range(n_descriptors):
            predictions[i] = clf.predict(descriptors[i].reshape(1, -1))

        # Component is signature if at least 50% of the descriptors
        # are classified as signature.
        n_votes = len(predictions)
        n_yes_v = n_votes - np.count_nonzero(predictions)
        confidence = n_yes_v / n_votes
        if confidence > 0.5 and confidence < 0.99:
            mask[idx] = 255

    return mask


def extract_signatures(dataset, outdir, model, preprocess=True):
    """Extracts signatures from all images in the dataset.

    Segmentation masks for each input image are generated and saved in the
    specified directory.

    Parameters:
        dataset (str) : Path of the test dataset.
        outdir (str) : Path to save the segmentation output.
        model (str) : Path of the extraction model to use.
        preprocess (bool) : Optional. Additional preprocessing happens before
                            feature extraction if True. Default is True.
    """
    # Load extraction model
    print("Loading segmentation model...")
    clf = joblib.load(model)

    # Get list of input files
    images = list_images(dataset)
    print("Found", len(images), "images. Starting segmentation...")

    # Create output directory if doesn't already exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for image_f in tqdm(images):
        im = cv2.imread(image_f, 0)
        mask = extract_signature(im, clf)

        outfile = os.path.split(image_f)[1]
        outfile = os.path.splitext(outfile)[0] + ".png"
        outfile = os.path.join(outdir, outfile)
        cv2.imwrite(outfile, mask)
