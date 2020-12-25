import os

from src import features, train, classify, list_images, jaccard_score, f1_score


# EXPERIMENTS
# components = ['bbdt8', 'sauf8', 'sauf4']
# descriptor = {'surf' : {'size:' [64, 128],
#                         'hessian:' 400,
#                         'octaves:' 4,
#                         'layers': 3},
#               'sift' : [],
#               'kaze' : [],
#               'akaze': []}
# classifier = {'decision-tree' : {'criteria': ['gini', 'entropy']}}
# id = 0

experiment = '03-bbdt8-surf-64-400'
output_dir = f'out/{experiment}'

# Extract features for training
train_dir = 'data/train/'
features_dir = f'{output_dir}/features/'
features(train_dir, features_dir)

# Train and save the classifer
models_dir = f'{output_dir}/models/'
criterion = 'entropy'
model_name = f'decision-tree-{criterion}'
train(features_dir, models_dir, model_name, criterion)
print(experiment, model_name)

# Classify test data
test_dir = 'data/test/'
temp_dir = f'{output_dir}/g-temp/'
classify(test_dir, temp_dir, os.path.join(models_dir, f'{model_name}.pkl'))

# Calculate accuracy of segmentation using Dice Coefficient (F1 Score)
predictions = f'{temp_dir}/{model_name}/'
groundtruth = 'data/groundtruth/'
dice = f1_score(predictions, groundtruth)
iou = jaccard_score(predictions, groundtruth)

print('F1 Score: %.02f%%' % (dice*100))
print('Jaccard Index: %.02f%%' % (iou*100))
