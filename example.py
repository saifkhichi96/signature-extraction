import os

from src import features, train, classify


# Extract features for training
train_dir = 'data/train/'
features_dir = 'out/features/'
features(train_dir, features_dir)

# Train and save the classifer
models_dir = 'out/models/'
model_name = 'decision-tree-01'
train(features_dir, models_dir, model_name)

# Classify test data
test_dir = 'data/test/'
out_dir = 'out/temp/'
classify(test_dir, out_dir, os.path.join(models_dir, f'{model_name}.pkl'))
