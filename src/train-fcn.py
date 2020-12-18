import datetime
import numpy as np
import os
import torch
import torchfcn
import torch.nn as nn

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid


class ChecksDataset(Dataset):
    """Bank checks dataset."""
    def __init__(self, root, transform=False):
        self.root = root
        self.mean = np.array([0, 0, 0])
        self.class_names = np.array(['bg', 'sign'])
        self._transform = transform

        self.files = []
        Xs = sorted(os.listdir(self.root + 'X'))
        ys = sorted(os.listdir(self.root + 'y'))
        for im in Xs:
            im = im[2:]
            if ('y_'+im) in ys:
                self.files.append({
                    'image': os.path.join(root, 'X/%s' % ('X_'+im)),
                    'label': os.path.join(root, 'y/%s' % ('y_'+im)),
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]

        # Load image
        image = Image.open(data_file['image']).convert('RGB')
        image = image.resize((512, 512))
        image = np.array(image, dtype=np.uint8)

        # Load label
        label = Image.open(data_file['label']).convert('1')
        label = label.resize((512, 512))
        label = np.array(label, dtype=np.int32)
        label[label == 255] = -1

        if self._transform:
            return self.transform(image, label)
        else:
            return image, label

    def transform(self, image, label):
        # Transform image
        image = image.astype(np.float64)        # uint8 -> float64
#         image -= self.mean                      # Remove mean
        image = np.transpose(image, (2, 0, 1))  # (W, H, C) -> (C, W, H)
        image = torch.from_numpy(image).float() # To tensor

        # Transform label
        label = torch.from_numpy(label).long()  # To tensor

        return image, label

    def untransform(self, image, label):
        # Restore image
        image = image.numpy()                   # To numpy array
        image = np.transpose(image, (1, 2, 0))  # (C, W, H) -> (W, H, C)
#         image += self.mean                      # Restore mean
        image = image.astype(np.uint8)          # float64 -> uint8

        # Restore label
        label = label.numpy()                   # To numpy array

        return image, label


# Load the Dataset
cuda = torch.cuda.is_available()
print('cuda:', cuda)

# Specify dataset input paths and create datasets
data_dir = '../data/bcsd/'

# Load datasets
print("Loading training set...")
train_data = ChecksDataset(data_dir + 'TrainSet/', transform=True)  # training set
print(len(train_data), "images in train set.\n")

print("Loading validation set...")
valid_data = ChecksDataset(data_dir + 'TestSet/', transform=True)    # validation set
print(len(valid_data), "images in validation set.\n")


# Create training and validation dataloaders.
print("Creating data loaders...")
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, **kwargs)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, **kwargs)

# Show some of the loaded images to verify dataloaders.
def convert_img(im, mask=False):
    print(im.shape)
    im = im.numpy()
    im = np.transpose(im, (1, 2, 0))
    if mask:
        im = im / np.max(im)

    return im

print("Show sample batch from created datasets...")
sample = iter(train_loader).next()

ax = plt.subplot(2, 1, 1)
plt.imshow(convert_img(make_grid(sample[0])))

ax = plt.subplot(2, 1, 2)
plt.imshow(convert_img(make_grid(sample[1]), mask=True))


# Create and train the Fully-Convolutional Network (FCN)

# Define training parameters
def get_parameters(model, bias=False):
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

# Create model
net = torchfcn.models.FCN8s(n_class=2)
if cuda:
    net = net.cuda()

# Define an optimizer
optim = torch.optim.SGD([{'params': get_parameters(net, bias=False)},
                         {'params': get_parameters(net, bias=True),
                          'lr': 1.0e-14 * 2, 'weight_decay': 0}],
                        lr=1.0e-14, momentum=0.99, weight_decay=0.0005)

# Create the trainer
logfile = os.path.join('../logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f'))
trainer = torchfcn.Trainer(
    cuda=cuda,
    model=net,
    optimizer=optim,
    train_loader=train_loader,
    val_loader=valid_loader,
    out=logfile,
    max_iter=5000,
    interval_validate=1000
)

# Start training
trainer.epoch = 0
trainer.iteration = 0
trainer.train()

# Validate
trainer.validate()
