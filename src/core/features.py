import os

import cv2
import numpy as np
from tqdm import tqdm

from utils import list_images
from .preprocess import remove_lines, threshold


def get_components(img, preprocess=False):
    """Finds and describes connected components in an image as SURF features.

    Grana's (BBDT) 8-way connectivity algorithm is used to find all the
    connected components in the image. Then, 128-bit extended SURF descriptors
    (as detailed in [1]) with a Hessian threshold of 400 are used to describe
    these components. These descriptors are the feature vectors.

    [1] Dengel, Andreas & Ahmed, Sheraz & Malik, Muhammad Imran & Liwicki,
    Marcus. (2012). Signature Segmentation from Document Images. Proceedings
    - International Workshop on Frontiers in Handwriting Recognition, IWFHR.
    10.1109/ICFHR.2012.271.

    Parameters:
        img (np.array) : A single-channel grayscale image as a numpy array.
        preprocess (bool) : Optional. If this is True, image is preprocessed
                            before feature extraction. Default is False.

    Returns:
        list of feature arrays for all detected connected components, and their
        locations in the image
    """
    # Binarize to get a two-color image
    img = threshold(img)

    # If specified, perform additional pre-processing steps
    if preprocess:
        # Based on our prior knowlegde that signature is usually found in the
        # bottom-right quarter of the bank check, make rest of image white
        h, w = img.shape
        img[:int(h / 2), :w] = 0
        img[:h, :int(w / 2)] = 0

        # Use a heuristic to try and remove horizontal lines from the image
        # This step is intended to remove the guidelines on bank checks
        img = remove_lines(img)

    # Find all connected components
    count, labels, stats, _ = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)

    # For each individual component, do
    components = []
    for idx in range(1, count):  # (0 is background, so ignore)
        # Crop out the component
        x, y, w, h = stats[idx, 0], stats[idx, 1], stats[idx, 2], stats[idx, 3]
        component = img[y: y + h, x: x + w]

        # Locate pixels belonging to the component
        idx = np.where(labels == idx)

        # Extract SURF features from the component
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400,
                                           nOctaves=4,
                                           nOctaveLayers=3,
                                           extended=True,
                                           upright=True)
        _, descriptors = surf.detectAndCompute(component, None)

        # Save descriptors and indices of the component
        if descriptors is not None:
            component = (np.array(descriptors), idx)
            components.append(component)

    return components


def extract(path, preprocess=False):
    """Extract features from all images in a directory.

    This function should be used to extract features for a single training class
    in the dataset.

    Parameters:
        path (str) : Location of the input directory.
        preprocess (bool) : Optional. If this is True, image is preprocessed
                            before feature extraction. Default is False.

    Returns:
        all extracted features as a Nx128 dimensional array, where N is the sum
        of number of all detected components in all images
    """
    components = None
    for image_f in tqdm(list_images(path)):
        try:
            # Open the image in OpenCV
            im = cv2.imread(image_f, 0)
            if im is None:
                raise IOError(f'{image_f} could not be opened.')

            for descriptors, idx in get_components(im, preprocess):
                component = np.vstack(descriptors)
                if components is None:
                    components = component
                else:
                    components = np.vstack((components, component))
        except Exception as e:
            print(e)

    return components


def extract_features(dataset, preprocess=False, out_dir=None):
    """Extract features from all images in a dataset.

    This function should be used to extract features for all classes in a
    dataset. The dataset must be organized such that, the dataset directory
    contains one subdirectory for each class, with each of these subdirectory
    containing images for that class.

    Classes will be sorted alphabetically and assigned labels [0,K-1], where K
    is the number of classes.

    Extracted features can optionally be stored by passing in a path to save
    the features in `out_dir` parameter. Features will be saved as arrays in .npy
    files with names same as the class name.

    Parameters:
        dataset (str) : Location of the input directory.
        preprocess (bool) : If this is True, image is preprocessed before
                            feature extraction. Default: False.
        out_dir (str|None) : Save path for the feature arrays. Default: None.

    Returns:
        The dataset as (X, y) tuple, where X is the feature array and y is the
        label vector.
    """
    # Get list of all training classes
    classes = [x for x in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, x))]
    print(f'Detected {len(classes)} Classes:', classes)

    X = []
    y = []
    for i, c in enumerate(sorted(classes)):
        # Extract features
        print(f'Extracting \'{c}\' features...')
        data = np.vstack(extract(os.path.join(dataset, c)))
        X.append(data)

        # Generate labels
        labels = np.ones((data.shape[0], 1)) * i
        y.append(labels)
        print(f'{data.shape[0]} feature vectors extracted')

        # Save features and labels if specified
        if out_dir is not None:
            print('Saving extracted features...')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        outfile = os.path.join(out_dir, f'{c}.npy')
        np.save(outfile, data)

    X = np.vstack(X)
    y = np.vstack(y).ravel()
    return X, y
