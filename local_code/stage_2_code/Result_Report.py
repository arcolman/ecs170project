'''
Results log, comparison table, and comparison plot for the report

results is a dict {config_name: scores}, where scores is the dict returned by Evaluate_Metrics.evaluate()
'''

from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'precision_weighted': 'Precision (weighted)', 'recall_weighted': 'Recall (weighted)', 'f1_weighted': 'F1 (weighted)',
    'precision_macro': 'Precision (macro)', 'recall_macro': 'Recall (macro)', 'f1_macro': 'F1 (macro)',
    'precision_micro': 'Precision (micro)', 'recall_micro': 'Recall (micro)', 'f1_micro': 'F1 (micro)',
}
# micro P/R/F1 equal accuracy for single-label multiclass, so they are left out of the default report view
REPORT_METRICS = ('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro')
PLOT_METRICS = ('accuracy', 'f1_weighted', 'f1_macro')
# colorblind-checked categorical colors, paired with distinct marker shapes so color is never the only cue
SERIES_STYLES = (('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', 'D'), ('#eda100', '^'))


def log_result(log_path, config_name, scores, config=None):
    '''Append one run to a JSON-lines log, so every tuning configuration and its scores is kept.'''
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {'name': config_name, 'config': config or {}, 'scores': {k: float(v) for k, v in scores.items()}}
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def load_results(log_path):
    '''Read a log written by log_result into {config_name: scores}. A re-run config keeps its latest scores.'''
    results = {}
    with open(log_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                results[entry['name']] = entry['scores']
    return results


def save_results_table(results, csv_path, metrics=REPORT_METRICS, sort_by='accuracy'):
    '''Write the comparison table as CSV and as a LaTeX tabular (same path, .tex), best config first.'''
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(results, key=lambda n: results[n][sort_by], reverse=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config'] + list(metrics))
        for name in names:
            writer.writerow([name] + [f'{results[name][m]:.4f}' for m in metrics])

    # bold the best score in each column
    best = {m: max(results[n][m] for n in names) for m in metrics}
    lines = ['\\begin{tabular}{l' + 'c' * len(metrics) + '}', '\\hline',
             'Configuration & ' + ' & '.join(METRIC_LABELS[m] for m in metrics) + ' \\\\', '\\hline']
    for name in names:
        cells = []
        for m in metrics:
            cell = f'{results[name][m]:.4f}'
            cells.append(f'\\textbf{{{cell}}}' if results[name][m] == best[m] else cell)
        lines.append(name.replace('_', '\\_') + ' & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\hline', '\\end{tabular}']
    csv_path.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    print(f'saved results table to {csv_path} and {csv_path.with_suffix(".tex")}')


def plot_comparison(results, fig_path, metrics=PLOT_METRICS, sort_by='accuracy'):
    '''Dot plot of each config's scores, best config on top. Dots (not bars) so the x-axis can zoom
    into the narrow range where MNIST-style scores differ, without bars misrepresenting magnitude.'''
    assert len(metrics) <= len(SERIES_STYLES), f'plot at most {len(SERIES_STYLES)} metrics; put the rest in the table'
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(results, key=lambda n: results[n][sort_by])
    y = range(len(names))

    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(names) + 1.4))
    # offset each metric vertically so near-identical scores (common here) do not hide each other
    offsets = [(i - (len(metrics) - 1) / 2) * 0.22 for i in range(len(metrics))]
    for m, (color, marker), dy in zip(metrics, SERIES_STYLES, offsets):
        ax.scatter([results[n][m] for n in names], [i - dy for i in y], color=color, marker=marker, s=45,
                   edgecolors='white', linewidths=1, zorder=3, label=METRIC_LABELS[m])

    values = [results[n][m] for n in names for m in metrics]
    pad = max((max(values) - min(values)) * 0.1, 0.005)
    ax.set_xlim(min(values) - pad, min(max(values) + pad, 1.0 + pad / 2))
    ax.set_yticks(list(y), names)
    ax.set_xlabel('Score on test set')
    ax.grid(axis='x', color='#e4e3df', linewidth=0.8, zorder=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(metrics), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f'saved comparison plot to {fig_path}')
