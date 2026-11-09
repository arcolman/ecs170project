"""Run from project root: python -m script.stage_3_script.script_cnn_cifar_select

Chooses the final CIFAR CNN among cifar_configs.CANDIDATES without touching the test set: holds out 5,000 of the
50,000 training images (500 per class) as a validation set, trains each candidate on the other 45,000 (with each
seed in cifar_configs.SEEDS), and ranks candidates by mean validation accuracy. Ties are broken by mean validation macro F1, then by fewer parameters.
The winner goes into cifar_configs.FINAL and is retrained on all 50,000 images by script_cnn_cifar.py.
Writes result/stage_3_result/cifar/selection/selection.csv.
"""
from pathlib import Path
import contextlib
import csv
import io
import numpy as np
from sklearn.model_selection import train_test_split
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_CIFAR
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from script.stage_3_script.cifar_configs import CANDIDATES, SEEDS


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / 'result' / 'stage_3_result' / 'cifar' / 'selection'
    out.mkdir(parents=True, exist_ok=True)

    data_obj = Dataset_Loader_CIFAR('CIFAR', '10-class object recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_3_data'
    train = data_obj.load()['train']
    # stratified, 5,000 images out of 50,000 with 10 classes = exactly 500 held-out images per class
    X_fit, X_val, y_fit, y_val = train_test_split(train['X'], train['y'], test_size=5000, stratify=train['y'],
                                                  random_state=0)
    assert np.bincount(y_val, minlength=10).tolist() == [500] * 10
    # the test set is deliberately replaced by the validation split here
    data = {'train': {'X': X_fit, 'y': y_fit}, 'test': {'X': X_val, 'y': y_val}}
    print(f'fit on {len(y_fit)}, validate on {len(y_val)} (test set unused)')

    rows = []
    for name, settings in CANDIDATES.items():
        val_acc, val_f1, times = [], [], []
        for seed in SEEDS:
            method_obj = Method_CNN_CIFAR(name, '', seed=seed, **settings)
            method_obj.data = data
            evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_obj.data = method_obj.run()
                scores = evaluate_obj.evaluate()
            val_acc.append(scores['accuracy'])
            val_f1.append(scores['f1_macro'])
            times.append(method_obj.method_training_time)
        n_params = sum(p.numel() for p in method_obj.parameters())
        rows.append((name, np.mean(val_acc), np.std(val_acc), np.mean(val_f1), n_params, np.mean(times)))
        print(f'{name}: validation accuracy {np.mean(val_acc):.4f} +/- {np.std(val_acc):.4f}', flush=True)

    rows.sort(key=lambda r: (-r[1], -r[3], r[4]))
    with open(out / 'selection.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate', 'val_accuracy_mean', 'val_accuracy_std', 'val_f1_macro_mean', 'n_parameters',
                         'training_time_s_mean'])
        for name, acc, std, f1, n, t in rows:
            writer.writerow([name, f'{acc:.4f}', f'{std:.4f}', f'{f1:.4f}', n, f'{t:.1f}'])
    print('************ Validation ranking ************')
    for name, acc, std, f1, n, t in rows:
        print(f'  {name:<28} val acc {acc:.4f} +/- {std:.4f}   val macro F1 {f1:.4f}   {n:>9,} params   {t:.0f}s')
    print(f'best: {rows[0][0]} -> set FINAL = CANDIDATES[{rows[0][0]!r}] in cifar_configs.py')


if __name__ == '__main__':
    main()
