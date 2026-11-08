"""Run from project root: python -m script.stage_3_script.script_cnn_orl_ablation [--fresh] [group ...]

ORL config experiments (spec step 3-5): trains every config in orl_configs.ABLATIONS with each seed in
orl_configs.SEEDS on the 360 training images and tests it on the 40 test images. Test accuracy is also recorded
after every epoch (monitor only, never used for training) to compare training and test accuracy.
Writes to result/stage_3_result/orl/ablation/:
  runs.jsonl                one line per (config, seed): scores, final training accuracy, per-epoch history
  ablation_table.csv/.tex   mean +/- std over the seeds for every config
  curves_<group>.png        training loss per epoch for each group (seed 1)
Resumes where it stopped if interrupted; --fresh starts over; group names run only those groups.
"""
from pathlib import Path
import warnings
import contextlib
import csv
import io
import json
import sys
import numpy as np
# 40 test images over 40 classes makes sklearn guess the labels might be a regression target; they are not
warnings.filterwarnings('ignore', message='The number of unique classes is greater than 50%')
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_ORL
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Result_Report import plot_learning_curves
from script.stage_3_script.orl_configs import ABLATIONS, BASELINE, SEEDS

TABLE_METRICS = ('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro')
TEX_LABELS = {'accuracy': 'Accuracy', 'precision_weighted': 'Precision (w)', 'recall_weighted': 'Recall (w)',
              'f1_weighted': 'F1 (w)', 'f1_macro': 'F1 (macro)', 'train_accuracy': 'Train acc.'}


def load_runs(path):
    runs = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                runs[(r['name'], r['seed'])] = r
    return runs


def train_one(name, settings, seed, data):
    method_obj = Method_CNN_ORL(name, '', seed=seed, monitor_data=data['test'], **settings)
    method_obj.data = data
    # the per-epoch log is long; keep the console to one line per run
    with contextlib.redirect_stdout(io.StringIO()):
        result = method_obj.run()
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    evaluate_obj.data = result
    with contextlib.redirect_stdout(io.StringIO()):
        scores = evaluate_obj.evaluate()
    return {'name': name, 'seed': seed, 'config': {k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()},
            'scores': {k: float(v) for k, v in scores.items()},
            'train_accuracy': float(method_obj.final_train_accuracy),
            'training_time_s': method_obj.method_training_time,
            'n_parameters': sum(p.numel() for p in method_obj.parameters()),
            'history': method_obj.history}


def summarize(runs, names):
    rows = []
    for name in names:
        rs = [r for (n, _), r in runs.items() if n == name]
        if not rs:
            continue
        row = {'name': name, 'n_seeds': len(rs), 'n_parameters': rs[0]['n_parameters'],
               'training_time_s': np.mean([r['training_time_s'] for r in rs])}
        for m in TABLE_METRICS:
            vals = [r['scores'][m] for r in rs]
            row[m] = (np.mean(vals), np.std(vals))
        vals = [r['train_accuracy'] for r in rs]
        row['train_accuracy'] = (np.mean(vals), np.std(vals))
        rows.append(row)
    return rows


def save_table(rows, out):
    cols = TABLE_METRICS + ('train_accuracy',)
    with open(out / 'ablation_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'n_seeds', 'n_parameters', 'training_time_s']
                   + [f'{c}_{s}' for c in cols for s in ('mean', 'std')])
        for r in rows:
            w.writerow([r['name'], r['n_seeds'], r['n_parameters'], f"{r['training_time_s']:.1f}"]
                       + [f'{v:.4f}' for c in cols for v in r[c]])
    lines = ['\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\hline',
             'Configuration & ' + ' & '.join(TEX_LABELS[c] for c in cols) + ' \\\\', '\\hline']
    for r in rows:
        lines.append(r['name'].replace('_', '\\_') + ' & '
                     + ' & '.join(f'{r[c][0]:.3f} $\\pm$ {r[c][1]:.3f}' for c in cols) + ' \\\\')
    lines += ['\\hline', '\\end{tabular}']
    (out / 'ablation_table.tex').write_text('\n'.join(lines) + '\n')


def main():
    args = [a for a in sys.argv[1:] if a != '--fresh']
    groups = args or list(ABLATIONS)
    root = Path(__file__).resolve().parents[2]
    out = root / 'result' / 'stage_3_result' / 'orl' / 'ablation'
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / 'runs.jsonl'
    if '--fresh' in sys.argv and runs_path.exists():
        runs_path.unlink()
    runs = load_runs(runs_path)

    data_obj = Dataset_Loader_ORL('ORL', '40-class face recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_3_data'
    data = data_obj.load()

    order = []
    for g in groups:
        for name, (_, changes) in ABLATIONS[g].items():
            if name not in order:
                order.append(name)
            for seed in SEEDS:
                if (name, seed) in runs:
                    continue
                r = train_one(name, {**BASELINE, **changes}, seed, data)
                runs[(name, seed)] = r
                with open(runs_path, 'a') as f:
                    f.write(json.dumps(r) + '\n')
                print(f"{g:<10} {name:<18} seed {seed}: test acc {r['scores']['accuracy']:.3f}  "
                      f"train acc {r['train_accuracy']:.3f}  final loss {r['history']['train_loss'][-1]:.4f}  "
                      f"{r['training_time_s']:.0f}s", flush=True)
        histories = {label: runs[(name, SEEDS[0])]['history'] for name, (label, _) in ABLATIONS[g].items()}
        with contextlib.redirect_stdout(io.StringIO()):
            plot_learning_curves(histories, out / f'curves_{g}.png', log_y=True,
                                 title=f'ORL CNN, {g}: training loss per epoch (seed {SEEDS[0]})')

    all_names = list(dict.fromkeys(n for g in ABLATIONS for n in ABLATIONS[g]))
    rows = summarize(runs, all_names)
    save_table(rows, out)
    print('************ ORL ablation (test set, mean +/- std over seeds) ************')
    for r in rows:
        print(f"  {r['name']:<18} acc {r['accuracy'][0]:.3f} +/- {r['accuracy'][1]:.3f}   "
              f"F1w {r['f1_weighted'][0]:.3f}   train acc {r['train_accuracy'][0]:.3f}   "
              f"{r['n_parameters']:>9,} params   {r['training_time_s']:.0f}s")


if __name__ == '__main__':
    main()
