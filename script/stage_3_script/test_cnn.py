"""Run from project root: python -m script.stage_3_script.test_cnn

Checks the Stage 3 loaders and Method_CNN_MNIST on small synthetic data, so it does not need the datasets.
"""
from pathlib import Path
import pickle
import tempfile
import numpy as np
import torch
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_MNIST, Dataset_Loader_ORL, Dataset_Loader_CIFAR
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST


def fake_pickle(folder, name, image_shape, labels):
    # same layout as the instructor files: {'train'/'test': [{'image': pixels, 'label': int}, ...]}
    rng = np.random.default_rng(0)
    split = lambda n: [{'image': rng.integers(0, 256, image_shape, dtype=np.uint8), 'label': int(l)} for l in labels[:n]]
    with open(Path(folder) / name, 'wb') as f:
        pickle.dump({'train': split(len(labels)), 'test': split(len(labels) // 2)}, f)


def check_loaders():
    with tempfile.TemporaryDirectory() as folder:
        fake_pickle(folder, 'MNIST', (28, 28), np.arange(20) % 10)
        # ORL: 3 identical channels, labels 1..40
        fake_pickle(folder, 'ORL', (112, 92, 3), np.arange(80) % 40 + 1)
        fake_pickle(folder, 'CIFAR', (32, 32, 3), np.arange(20) % 10)
        for loader_class, shape, classes in ((Dataset_Loader_MNIST, (1, 28, 28), 10), (Dataset_Loader_ORL, (1, 112, 92), 40),
                                             (Dataset_Loader_CIFAR, (3, 32, 32), 10)):
            loader = loader_class('', '')
            loader.dataset_source_folder_path = folder
            data = loader.load()
            for d in data.values():
                assert d['X'].shape[1:] == shape and d['X'].dtype == np.float32
                assert 0 <= d['X'].min() <= d['X'].max() <= 1
                assert d['y'].min() == 0 and d['y'].max() < classes
            assert data['train']['y'].max() == classes - 1


def synthetic_images(n, seed):
    # 10 classes, each a bright 7x7 square at its own position on a noisy 28x28 image
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, size=n)
    X = rng.random((n, 1, 28, 28), dtype=np.float32) * 0.3
    for i, label in enumerate(y):
        r, c = divmod(label, 5)
        X[i, 0, 3 + 12 * r:10 + 12 * r, 1 + 5 * c:8 + 5 * c] = 1.0
    return X, y.astype(np.int64)


def check_cnn():
    X_train, y_train = synthetic_images(1000, seed=1)
    X_test, y_test = synthetic_images(300, seed=2)
    data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

    # output portal: 10 logits per image, for every architecture variant
    for settings in ({}, {'conv_channels': (8,), 'kernel_size': 3}, {'conv_channels': (8, 8, 8), 'pool': 'avg'},
                     {'batch_norm': True, 'fc_dims': (32,), 'dropout': 0.5, 'stride': 2}, {'padding': 0}):
        assert Method_CNN_MNIST('cnn', '', **settings).forward(torch.zeros(5, 1, 28, 28)).shape == (5, 10)
    # other input sizes (ORL / CIFAR copies) only need the portal settings changed
    assert Method_CNN_MNIST('cnn', '', in_channels=3, input_size=(32, 32)).forward(torch.zeros(2, 3, 32, 32)).shape == (2, 10)

    model = Method_CNN_MNIST('cnn', '', conv_channels=(8, 16), max_epoch=4, batch_size=64)
    model.data = data
    result = model.run()
    accuracy = (result['pred_y'].numpy() == y_test).mean()
    print(f'loss {model.history["train_loss"][0]:.3f} -> {model.history["train_loss"][-1]:.3f}, test accuracy {accuracy:.3f}')
    assert model.history['train_loss'][-1] < model.history['train_loss'][0]
    assert accuracy > 0.9

    try:
        Method_CNN_MNIST('cnn', '', kernal_size=3)
        raise AssertionError('unknown setting was accepted')
    except ValueError:
        pass


def main():
    check_loaders()
    check_cnn()
    print('Stage 3 checks passed.')


if __name__ == '__main__':
    main()
