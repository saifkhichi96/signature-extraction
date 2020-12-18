import json
import os
import time

import joblib
import numpy as np
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
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

if __name__ == '__main__':
    with open('params.json') as json_file:
        data = json.load(json_file)

        # Set dataset path
        data_dir = os.path.join(data['output'], 'features/')

        # Set output path (for saving trained features)
        save_dir = os.path.join(data['output'], f'models/{time.time()}/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Load the dataset
        print('Reading the dataset... ')
        X, y = load_dataset(data_dir)
        assert X.shape[0] == y.shape[0]
        print(f'{X.shape[0]} samples found\n')

        # Create models to be trained
        models = {
            'mlp' : MLPClassifier(alpha=1),
            'sgd' : SGDClassifier(),
            'knn' : KNeighborsClassifier(),
            'svc-rbf' : SVC(kernel='rbf'),
            'svc-linear' : LinearSVC(),
            'decision-tree' : DecisionTreeClassifier(),
        }

        # Train all models one-by-one
        for name, clf in models.items():
            print(f'Training {name}... ')
            start = time.time()
            clf.fit(X, y)
            print(f'{time.time() - start} seconds\n')

            print(f'Saving trained model... ')
            start = time.time()
            outfile = os.path.join(save_dir, f'{name}.pkl')
            joblib.dump(clf, outfile)
            print(f'{time.time() - start} seconds\n\n')
