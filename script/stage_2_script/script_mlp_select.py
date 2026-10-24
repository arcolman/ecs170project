"""Run from project root: python -m script.stage_2_script.script_mlp_select

Chooses the final MLP among mlp_configs.CANDIDATES without touching the test set: holds out a stratified 10,000 of
the 60,000 training instances as a validation set, trains each candidate on the other 50,000 (with each seed in
SELECT_SEEDS), and ranks candidates by mean validation accuracy. The winner then goes into mlp_configs.FINAL and is
retrained on all 60,000 by script_mlp.py, which reports its test scores.
Writes result/stage_2_result/selection/selection.csv.
"""
from pathlib import Path
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from script.stage_2_script.mlp_configs import CANDIDATES

SELECT_SEEDS = (1, 2)
VALIDATION_SIZE = 10000


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / 'result' / 'stage_2_result' / 'selection'
    out.mkdir(parents=True, exist_ok=True)

    data_obj = Dataset_Loader('handwritten digits', '10-class digit recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_2_data'
    train = data_obj.load()['train']
    X_fit, X_val, y_fit, y_val = train_test_split(train['X'], train['y'], test_size=VALIDATION_SIZE,
                                                  stratify=train['y'], random_state=0)
    # the test set is deliberately replaced by the validation split here
    data = {'train': {'X': X_fit, 'y': y_fit}, 'test': {'X': X_val, 'y': y_val}}
    print(f'fit on {len(y_fit)}, validate on {len(y_val)} (test set unused)')

    rows = []
    for name, settings in CANDIDATES.items():
        val_acc, val_f1, times = [], [], []
        for seed in SELECT_SEEDS:
            print(f'************ {name}, seed {seed} ************')
            method_obj = Method_MLP(name, '', seed=seed, **settings)
            method_obj.data = data
            evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
            evaluate_obj.data = method_obj.run()
            scores = evaluate_obj.evaluate()
            val_acc.append(scores['accuracy'])
            val_f1.append(scores['f1_macro'])
            times.append(method_obj.method_training_time)
        rows.append((name, np.mean(val_acc), np.std(val_acc), np.mean(val_f1), np.mean(times)))
        print(f'{name}: validation accuracy {np.mean(val_acc):.4f} +/- {np.std(val_acc):.4f}')

    rows.sort(key=lambda r: r[1], reverse=True)
    with open(out / 'selection.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate', 'val_accuracy_mean', 'val_accuracy_std', 'val_f1_macro_mean', 'training_time_s_mean'])
        for name, acc, std, f1, t in rows:
            writer.writerow([name, f'{acc:.4f}', f'{std:.4f}', f'{f1:.4f}', f'{t:.1f}'])
    print('************ Validation ranking ************')
    for name, acc, std, f1, t in rows:
        print(f'  {name:<28} val acc {acc:.4f} +/- {std:.4f}   val macro F1 {f1:.4f}   {t:.0f}s')
    print(f'best: {rows[0][0]} -> set FINAL = CANDIDATES[{rows[0][0]!r}] in mlp_configs.py')


if __name__ == '__main__':
    main()
