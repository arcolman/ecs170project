"""Load the instructor-provided Stage 3 image datasets (MNIST, ORL, CIFAR).

Each dataset is one pickle file in data/stage_3_data/ holding {'train': [...], 'test': [...]}, where every instance is
{'image': H x W (x 3) pixel matrix, 'label': int}. The given train/test split is kept as is.
load() returns {'train': {'X', 'y'}, 'test': {'X', 'y'}} with X a float32 (N, C, H, W) array scaled to [0, 1]
and y an int64 array of labels starting at 0, ready for nn.Conv2d and nn.CrossEntropyLoss.
"""
from pathlib import Path
import pickle
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader_Image(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None
    # set by each subclass: the expected (C, H, W) of one image and the number of classes
    image_shape = None
    num_classes = None

    def to_chw(self, images):
        '''(N, H, W) or (N, H, W, 3) pixels -> (N, C, H, W)'''
        if images.ndim == 3:
            return images[:, None, :, :]
        return images.transpose(0, 3, 1, 2)

    def load(self):
        path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        print(f'loading data from {path}...')
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        labels = {split: np.array([instance['label'] for instance in raw[split]], dtype=np.int64) for split in ('train', 'test')}
        # shift both splits by the same offset (ORL is labelled 1..40, CrossEntropyLoss wants 0..39)
        offset = min(labels['train'].min(), labels['test'].min())
        data = {}
        for split in ('train', 'test'):
            images = np.stack([np.asarray(instance['image']) for instance in raw[split]])
            X = self.to_chw(images).astype(np.float32) / 255.0
            y = labels[split] - offset
            data[split] = {'X': X, 'y': y}
            if X.shape[1:] != self.image_shape:
                raise ValueError(f'{path} {split}: expected images of shape {self.image_shape}, got {X.shape[1:]}')
            if y.min() < 0 or y.max() >= self.num_classes:
                raise ValueError(f'{path} {split}: labels must be between 0 and {self.num_classes - 1}, got {y.min()}..{y.max()}')
            print(f'{split}: X={X.shape}, y={y.shape}, labels {y.min()}..{y.max()}')
        return data


class Dataset_Loader_MNIST(Dataset_Loader_Image):
    # 28x28 grayscale handwritten digits, 60,000 train / 10,000 test
    dataset_source_file_name = 'MNIST'
    image_shape = (1, 28, 28)
    num_classes = 10


class Dataset_Loader_ORL(Dataset_Loader_Image):
    # 112x92 face images of 40 people, 9 train / 1 test per person
    dataset_source_file_name = 'ORL'
    image_shape = (1, 112, 92)
    num_classes = 40

    def to_chw(self, images):
        # stored as RGB with R = G = B, so one channel carries all the information
        return images[:, None, :, :, 0] if images.ndim == 4 else images[:, None, :, :]


class Dataset_Loader_CIFAR(Dataset_Loader_Image):
    # 32x32 color objects in 10 classes, 50,000 train / 10,000 test
    dataset_source_file_name = 'CIFAR'
    image_shape = (3, 32, 32)
    num_classes = 10
