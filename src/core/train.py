import os

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def load_dataset(dataset):
    """Loads dataset from given path.

    Parameters:
        dataset (str) : Path of the dataset.

    Returns:
        The dataset as (data, labels) tuple, where 'data' is the feature array
        and 'labels' is the label vector.
    """
    ext = '.npy'
    classes = [x.split('.')[0] for x in os.listdir(dataset) if x.lower().endswith(ext)]

    data = []
    labels = []
    for idx, cls in enumerate(sorted(classes)):
        data = np.load(os.path.join(dataset, f'{cls}{ext}'))
        data.append(data)

        labels = np.ones((data.shape[0], 1)) * idx
        labels.append(labels)

    data = np.vstack(data)
    labels = np.vstack(labels).ravel()
    return data, labels


def train(dataset, outfile):
    """Trains a classifier on the given dataset.

    A Decision Tree classifier with entropy criterion is trained on the dataset
    to distinguish between signature and non-signature components. Optionally,
    the trained model can also be saved to a file.

    Parameters:
        dataset (str) : Path of the dataset. This folder must contain extracted
                        features for each class in the dataset, saved as
                        separate .npy files.
        outfile (str) : Optional. Save path for the trained model.

    Returns:
        The classifier trained on given dataset.
    """
    # Load the dataset
    print('Reading the dataset... ')
    data, labels = load_dataset(dataset)
    assert data.shape[0] == labels.shape[0]
    print(f'{data.shape[0]} samples found\n')

    # Train a decision tree classifier
    print(f'Training the classifier...')
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(data, labels)

    if outfile is not None:
        print(f'Saving trained model...')
        out_dir = os.path.dirname(outfile)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        joblib.dump(clf, outfile)

    return clf
