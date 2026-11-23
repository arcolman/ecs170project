"""Run from project root: python -m script.stage_4_script.plot_cls_comparison

One figure comparing the vanilla RNN, LSTM and GRU classifiers (same settings, only the recurrent unit differs):
validation loss (left) and validation accuracy (right) per epoch, with each model's selected (best validation
accuracy) epoch marked. Reads <run>_history.json written by script_rnn_cls and saves
result/stage_4_result/classification/cls_rnn_lstm_gru_val_curves.png.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from local_code.stage_4_code.Result_Report import SERIES_STYLES

RUNS = (('rnn', 'Vanilla RNN'), ('lstm', 'LSTM'), ('gru', 'GRU'))

folder = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result' / 'classification'
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
for (run, label), (color, marker) in zip(RUNS, SERIES_STYLES):
    history = json.loads((folder / f'{run}_history.json').read_text())
    epochs = history['epoch']
    # same rule as restore_best: first epoch with the highest validation accuracy
    best = max(range(len(epochs)), key=lambda i: (history['val_accuracy'][i], -i))
    for ax, key in zip(axes, ('val_loss', 'val_accuracy')):
        ax.plot(epochs, history[key], color=color, linewidth=2, marker=marker, markersize=5,
                markeredgecolor='white', label=label, zorder=3)
        ax.plot(epochs[best], history[key][best], marker=marker, markersize=11, markerfacecolor='none',
                markeredgecolor=color, markeredgewidth=1.5, zorder=4)
for ax, ylabel in zip(axes, ('Validation cross-entropy loss', 'Validation accuracy')):
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(color='#e4e3df', linewidth=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[1].legend(frameon=False, fontsize=9)
fig.suptitle('IMDB validation curves (large open marker = selected epoch)', fontsize=10, x=0.01, ha='left')
fig.tight_layout()
out = folder / 'cls_rnn_lstm_gru_val_curves.png'
fig.savefig(out, dpi=200)
plt.close(fig)
print(f'saved comparison plot to {out}')
