import cv2
import numpy as np
import os

from tqdm import tqdm

from .preprocess import remove_lines, otsu
from .utils import list_images


def describe_image(image_f, connectivity=8, preprocess=False):
    """Describes an image in terms of features[1] of its connected components.

    [1] Dengel, Andreas & Ahmed, Sheraz & Malik, Muhammad Imran & Liwicki,
    Marcus. (2012). Signature Segmentation from Document Images. Proceedings
    - International Workshop on Frontiers in Handwriting Recognition, IWFHR.
    10.1109/ICFHR.2012.271.

    Parameters:
        image_f (str) : Path of the image file from which to extract components.
        connectivity (int) : Components can be extracted with either a 4-way or
                             8-way connectivity. Default is 8.
        preprocess (bool) : If this is True, image is prerocessed before feature
                            extraction. Default is False.

    Returns:
        list of feature arrays for all detected connected components
    """
    # Open the image in OpenCV
    im = cv2.imread(image_f, 0)
    if im is None:
        raise IOError(f'{image_f} could not be opened.')

    # Pre-process the image before extracting components
    if preprocess:
        # Remove top and left halves of the image
        h, w = im.shape
        im[:int(h/2), :w] = np.mean(im[:int(h/2), :w])
        im[:h, :int(w/2)] = np.mean(im[:h, :int(w/2)])

    im = otsu(im)
    if preprocess:
        im = remove_lines(im)

    # Find all connected components
    count, labels, stats, _ = cv2.connectedComponentsWithStats(im, connectivity, cv2.CV_32S)

    components = []
    for id in range(1, count): # skip 0, it is background
        # Crop out the component
        x, y, w, h = stats[id, 0], stats[id, 1], stats[id, 2], stats[id, 3]
        component = im[y : y + h, x : x + w]

        # Locate pixels belonging to the component
        idx = np.where(labels == id)

        # Extract SURF features from the component
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400,
                                           nOctaves=4,
                                           nOctaveLayers=3,
                                           extended=True,
                                           upright=True)
        _, descriptors = surf.detectAndCompute(component, None)

        # Save descriptors and location of the component
        if descriptors is not None:
            components.append([np.array(descriptors), idx])

    return components


def extract_features(path):
    """Extract features from all images in a directory.

    Parameters:
        path (str) : Location of the input directory.

    Returns:
        list of all feature arrays
    """
    components = None
    for im in tqdm(list_images(path)):
        try:
            for c, _ in describe_image(im):
                comp = np.vstack(c)
                if components is None:
                    components = comp
                else:
                    components = np.vstack((components, comp))
        except Exception as e:
            print(e)

    return components


def features(train_dir, save_dir):
    # Get list of all training classes
    classes = [x for x in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, x))]
    print(f'Detected {len(classes)} Classes:', classes)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for cls in classes:
        print(f'Extracting \'{cls}\' features...')
        features = np.vstack(extract_features(os.path.join(train_dir, cls)))
        print(f'{features.shape[0]} extracted')

        print('Saving extracted features...')
        outfile = os.path.join(save_dir, f'{cls}.npy')
        np.save(outfile, features)
