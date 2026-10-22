"""Load the instructor-provided Stage 2 CSV train/test datasets."""
from pathlib import Path
import numpy as np
from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    dataset_source_folder_path = None

    def load(self):
        folder = Path(self.dataset_source_folder_path)
        data = {}
        for split in ('train', 'test'):
            path = folder / f'{split}.csv'
            raw = np.loadtxt(path, delimiter=',', dtype=np.uint8)
            if raw.ndim != 2 or raw.shape[1] != 785:
                raise ValueError(f'{path}: expected 785 columns, got {raw.shape}')
            labels = raw[:, 0].astype(np.int64)
            if np.any(labels > 9):
                raise ValueError(f'{path}: labels must be between 0 and 9')
            features = raw[:, 1:].astype(np.float32) / 255.0
            data[split] = {'X': features, 'y': labels}
            print(f'{split}: X={features.shape}, y={labels.shape}, class counts={np.bincount(labels, minlength=10).tolist()}')
        return data
