import os

from core import extract_features, extract_signatures, train
from utils import list_images, jaccard_score, f1_score


train_data = 'data/train/'
valid_data = 'data/test/y/'

feats = 'out/features/'
tmp = 'out/temp/'
model = 'out/models/decision-tree.pkl'

# Extract features, train model and segment test data
extract_features(train_data, preprocess=False, outdir=feats)
train(feats, model)
extract_signatures(valid_data, tmp, model)

# Calculate accuracy of segmentation
predictions = f'out/temp/'
groundtruth = 'data/test/y_true'
dice = f1_score(predictions, groundtruth)
iou = jaccard_score(predictions, groundtruth)

print('F1 Score: %.02f%%' % (dice*100))
print('Jaccard Index: %.02f%%' % (iou*100))
