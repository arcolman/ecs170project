"""Run from project root: python -m script.stage_3_script.check_data

Loads MNIST, ORL and CIFAR from data/stage_3_data/ and prints each split's shape, pixel range, label range and class
counts. Saves one sample image per dataset and this printout to result/stage_3_result/data_check/.
"""
from pathlib import Path
import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_MNIST, Dataset_Loader_ORL, Dataset_Loader_CIFAR

LOADERS = {'MNIST': Dataset_Loader_MNIST, 'ORL': Dataset_Loader_ORL, 'CIFAR': Dataset_Loader_CIFAR}


def check(name, loader_class, data_folder, out_folder):
    loader = loader_class(name, '')
    loader.dataset_source_folder_path = data_folder
    data = loader.load()
    for split, d in data.items():
        X, y = d['X'], d['y']
        print(f'{name} {split}: X {X.shape} {X.dtype}, pixels {X.min():.2f}..{X.max():.2f}, '
              f'labels {y.min()}..{y.max()}, class counts {np.bincount(y).tolist()}')
        assert X.ndim == 4 and X.shape[1:] == loader.image_shape
        assert 0 <= X.min() <= X.max() <= 1
        assert 0 <= y.min() and y.max() < loader.num_classes

    image, label = data['train']['X'][0], data['train']['y'][0]
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    # (C, H, W) -> (H, W) for grayscale, (H, W, 3) for color
    ax.imshow(image[0] if image.shape[0] == 1 else image.transpose(1, 2, 0), cmap='gray')
    ax.set_title(f'{name} train[0], label {label}', fontsize=9)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_folder / f'{name.lower()}_sample.png', dpi=150)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    data_folder = root / 'data' / 'stage_3_data'
    out_folder = root / 'result' / 'stage_3_result' / 'data_check'
    out_folder.mkdir(parents=True, exist_ok=True)

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for name, loader_class in LOADERS.items():
            check(name, loader_class, data_folder, out_folder)
        print('Dataset checks passed. Ready for CNN training.')
    print(log.getvalue(), end='')
    (out_folder / 'check_data_output.txt').write_text(log.getvalue())
    print(f'saved output and sample images to {out_folder}')


if __name__ == '__main__':
    main()
