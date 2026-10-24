"""Run from project root: python -m script.stage_2_script.script_mlp_ablation

Ablation study for the report (Section 3.7): trains every configuration in mlp_configs.ABLATIONS with each seed in
mlp_configs.SEEDS, scores it on the test set, and writes to result/stage_2_result/ablation/:
  runs.jsonl                  one line per (config, seed) run: settings, scores, per-epoch history, timings
  ablation_table.csv / .tex   mean +/- std over seeds for every config, grouped by the setting that was changed
  curves_<group>.png          training loss curves for each group (first seed)
It also logs each config's mean scores to results_log.jsonl, next to the decision tree reference.

Finished runs are skipped when the script is re-run, so an interrupted sweep picks up where it stopped.
Pass --fresh to discard earlier runs, or group names (e.g. depth optimizer) to run only those groups.
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Result_Report import log_result, load_results, save_results_table, plot_comparison, plot_learning_curves, METRIC_LABELS, REPORT_METRICS
from script.stage_2_script.mlp_configs import ABLATIONS, BASELINE, SEEDS


def run_key(name, seed, settings):
    # settings are part of the key, so editing a config in mlp_configs.py re-runs it instead of reusing stale scores
    return json.dumps([name, seed, settings], sort_keys=True, default=list)


def load_runs(runs_path):
    runs = {}
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            if line.strip():
                run = json.loads(line)
                runs[run_key(run['name'], run['seed'], run['settings'])] = run
    return runs


def train_one(data, name, seed, settings):
    method_obj = Method_MLP(name, '', seed=seed, **settings)
    method_obj.data = data
    learned_result = method_obj.run()
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    evaluate_obj.data = learned_result
    scores = evaluate_obj.evaluate()
    return {'name': name, 'seed': seed, 'settings': settings, 'scores': {k: float(v) for k, v in scores.items()},
            'final_train_accuracy': float(method_obj.final_train_accuracy), 'history': method_obj.history,
            'n_parameters': sum(p.numel() for p in method_obj.parameters()),
            'training_time_s': method_obj.method_training_time}


def summarize(runs_for_config):
    '''mean and std over seeds of every score'''
    metrics = runs_for_config[0]['scores'].keys()
    mean = {m: float(np.mean([r['scores'][m] for r in runs_for_config])) for m in metrics}
    std = {m: float(np.std([r['scores'][m] for r in runs_for_config])) for m in metrics}
    return mean, std


def save_ablation_table(groups, summaries, out):
    rows = []
    for group, configs in groups.items():
        for name, (label, _) in configs.items():
            mean, std, extra = summaries[name]
            rows.append((group, name, label, mean, std, extra))

    with open(out / 'ablation_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'config', 'label'] + [f'{m}_{s}' for m in REPORT_METRICS for s in ('mean', 'std')]
                        + ['final_train_accuracy_mean', 'n_parameters', 'training_time_s_mean'])
        for group, name, label, mean, std, extra in rows:
            writer.writerow([group, name, label] + [f'{v:.4f}' for m in REPORT_METRICS for v in (mean[m], std[m])]
                            + [f'{extra["final_train_accuracy"]:.4f}', extra['n_parameters'], f'{extra["training_time_s"]:.1f}'])

    # LaTeX: one block per group, best mean accuracy in each group in bold, scores as percentages to save width
    metrics = ('accuracy', 'f1_weighted', 'f1_macro')
    lines = ['\\begin{tabular}{ll' + 'c' * len(metrics) + '}', '\\hline',
             'Changed setting & Value & ' + ' & '.join(METRIC_LABELS[m] + ' (\\%)' for m in metrics) + ' \\\\', '\\hline']
    for group, configs in groups.items():
        best = max(configs, key=lambda n: summaries[n][0]['accuracy'])
        for i, (name, (label, _)) in enumerate(configs.items()):
            mean, std, _ = summaries[name]
            cells = [f'{100 * mean[m]:.2f} $\\pm$ {100 * std[m]:.2f}' for m in metrics]
            if name == best:
                cells = [f'\\textbf{{{c}}}' for c in cells]
            group_cell = group.replace('_', ' ') if i == 0 else ''
            lines.append(f'{group_cell} & {label.replace("_", " ")} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\hline')
    lines.append('\\end{tabular}')
    (out / 'ablation_table.tex').write_text('\n'.join(lines) + '\n')
    print(f'saved ablation table to {out / "ablation_table.csv"} and .tex')


def main():
    args = sys.argv[1:]
    root = Path(__file__).resolve().parents[2]
    result_folder = root / 'result' / 'stage_2_result'
    out = result_folder / 'ablation'
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / 'runs.jsonl'
    if '--fresh' in args:
        runs_path.unlink(missing_ok=True)
    selected = [a for a in args if not a.startswith('--')]
    unknown = set(selected) - set(ABLATIONS)
    if unknown:
        raise SystemExit(f'unknown groups {sorted(unknown)}; choose from {list(ABLATIONS)}')
    groups = {g: c for g, c in ABLATIONS.items() if not selected or g in selected}

    data_obj = Dataset_Loader('handwritten digits', '10-class digit recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_2_data'
    data = data_obj.load()

    runs = load_runs(runs_path)
    configs = {name: {**BASELINE, **overrides} for g in groups.values() for name, (_, overrides) in g.items()}
    todo = [(n, s) for n in configs for s in SEEDS if run_key(n, s, configs[n]) not in runs]
    print(f'{len(configs)} configs x {len(SEEDS)} seeds, {len(todo)} runs to do')
    for i, (name, seed) in enumerate(todo, 1):
        print(f'************ [{i}/{len(todo)}] {name}, seed {seed} ************')
        run = train_one(data, name, seed, configs[name])
        runs[run_key(name, seed, configs[name])] = run
        with open(runs_path, 'a') as f:
            f.write(json.dumps(run, default=list) + '\n')
        print(f'test accuracy {run["scores"]["accuracy"]:.4f}, {run["training_time_s"]:.1f}s')

    summaries = {}
    log_path = result_folder / 'results_log.jsonl'
    for name, settings in configs.items():
        config_runs = [runs[run_key(name, s, settings)] for s in SEEDS]
        mean, std = summarize(config_runs)
        extra = {'final_train_accuracy': float(np.mean([r['final_train_accuracy'] for r in config_runs])),
                 'n_parameters': config_runs[0]['n_parameters'],
                 'training_time_s': float(np.mean([r['training_time_s'] for r in config_runs]))}
        summaries[name] = (mean, std, extra)
        log_result(log_path, f'mlp_{name}', mean, {**settings, 'seeds': list(SEEDS), 'std': std})

    save_ablation_table(groups, summaries, out)
    first_seed = SEEDS[0]
    for group, group_configs in groups.items():
        histories = {label: runs[run_key(name, first_seed, configs[name])]['history'] for name, (label, _) in group_configs.items()}
        plot_learning_curves(histories, out / f'curves_{group}.png', log_y=True,
                             title=f'Training loss by {group.replace("_", " ")} (seed {first_seed})')

    results = load_results(log_path)
    save_results_table(results, result_folder / 'results_table.csv')
    plot_comparison(results, result_folder / 'results_comparison.png')

    print('************ Ablation summary (mean over seeds) ************')
    for group, group_configs in groups.items():
        print(f'--- {group}')
        for name, (label, _) in group_configs.items():
            mean, std, extra = summaries[name]
            print(f'  {label:<45} acc {mean["accuracy"]:.4f} +/- {std["accuracy"]:.4f}   '
                  f'macro F1 {mean["f1_macro"]:.4f}   train acc {extra["final_train_accuracy"]:.4f}   {extra["training_time_s"]:.0f}s')


if __name__ == '__main__':
    main()
