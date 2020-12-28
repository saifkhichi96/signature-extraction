import joblib
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier


def load_dataset(dataset):
    """Loads dataset from given path.

    Parameters:
        dataset (str) : Path of the dataset.

    Returns:
        The dataset as (X, y) tuple, where X is the feature array and y is the
        label vector.
    """
    ext = '.npy'
    classes = [x.split('.')[0] for x in os.listdir(dataset) if x.lower().endswith(ext)]

    X = []
    y = []
    for idx, cls in enumerate(sorted(classes)):
        data = np.load(os.path.join(dataset, f'{cls}{ext}'))
        X.append(data)

        labels = np.ones((data.shape[0], 1)) * idx
        y.append(labels)

    X = np.vstack(X)
    y = np.vstack(y).ravel()
    return X, y

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
    X, y = load_dataset(dataset)
    assert X.shape[0] == y.shape[0]
    print(f'{X.shape[0]} samples found\n')

    # Train a decision tree classifier
    print(f'Training the classifier...')
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)

    if outfile is not None:
        print(f'Saving trained model...')
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        joblib.dump(clf, outfile)

    return clf
