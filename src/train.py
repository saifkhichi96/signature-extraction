import joblib
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier


def load_dataset(data_dir):
    ext = '.npy'
    classes = [x.split('.')[0] for x in os.listdir(data_dir) if x.lower().endswith(ext)]

    X = []
    y = []
    for idx, cls in enumerate(classes):
        data = np.load(os.path.join(data_dir, f'{cls}{ext}'))
        X.append(data)

        labels = np.ones((data.shape[0], 1)) * idx
        y.append(labels)

    X = np.vstack(X)
    y = np.vstack(y).ravel()
    return X, y

def train(data_dir, save_dir, outfile, criterion):
    # Load the dataset
    print('Reading the dataset... ')
    X, y = load_dataset(data_dir)
    assert X.shape[0] == y.shape[0]
    print(f'{X.shape[0]} samples found\n')

    # Train a decision tree classifier
    print(f'Training classifier...')
    clf = DecisionTreeClassifier(criterion=criterion)
    clf.fit(X, y)

    print(f'Saving trained model...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    outfile = os.path.join(save_dir, f'{outfile}.pkl')
    joblib.dump(clf, outfile)
