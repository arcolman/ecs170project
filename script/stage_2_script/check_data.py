"""Run from project root: python -m script.stage_2_script.check_data"""
from pathlib import Path
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader

def main():
    root = Path(__file__).resolve().parents[2]
    loader = Dataset_Loader('handwritten digits', '10-class digit recognition')
    loader.dataset_source_folder_path = root / 'data' / 'stage_2_data'
    data = loader.load()
    assert data['train']['X'].shape == (60000, 784)
    assert data['test']['X'].shape == (10000, 784)
    assert all(0 <= d['X'].min() <= d['X'].max() <= 1 for d in data.values())
    print('Dataset checks passed. Ready for MLP integration.')

if __name__ == '__main__':
    main()
