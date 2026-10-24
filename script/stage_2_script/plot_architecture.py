"""Run from project root: python -m script.stage_2_script.plot_architecture

Draws the final MLP architecture (mlp_configs.FINAL) for the report (Section 2) to
result/stage_2_result/mlp_architecture.png.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from script.stage_2_script.mlp_configs import FINAL

INK, MUTED, EDGE = '#1f1f1d', '#6b6a66', '#8a8984'
FILLS = {'input': '#e8f1fb', 'hidden': '#2a78d6', 'output': '#eb6834', 'loss': '#f4f3f0'}
ACTIVATION_NAMES = {'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid', 'leaky_relu': 'LeakyReLU', 'gelu': 'GELU'}


def main():
    root = Path(__file__).resolve().parents[2]
    fig_path = root / 'result' / 'stage_2_result' / 'mlp_architecture.png'
    hidden = list(FINAL['hidden_dims'])
    act = ACTIVATION_NAMES[FINAL['activation']]
    dropout = FINAL.get('dropout', 0.0)

    # one column per stage: input, each hidden layer, output logits, softmax + loss / prediction
    stages = [('input', 'Input', '784 pixels\n(28 x 28, scaled to [0,1])')]
    prev = 784
    for i, h in enumerate(hidden, 1):
        detail = f'Linear {prev}→{h}\n{act}' + (f'\nDropout p={dropout}' if dropout else '')
        stages.append(('hidden', f'Hidden {i}\n{h} units', detail))
        prev = h
    stages.append(('output', 'Output\n10 logits', f'Linear {prev}→10\n(no activation)'))
    smoothing = FINAL.get('label_smoothing', 0.0)
    loss_text = 'train: cross-entropy loss' + (f'\n(label smoothing {smoothing})' if smoothing else '')
    stages.append(('loss', 'Softmax', loss_text + '\ntest: argmax = digit 0-9'))

    n = len(stages)
    fig, ax = plt.subplots(figsize=(1.75 * n, 2.1))
    ax.set_xlim(0, n)
    ax.set_ylim(0.2, 0.88)
    ax.axis('off')
    box_w, box_h, box_y = 0.78, 0.36, 0.48
    for i, (kind, title, detail) in enumerate(stages):
        x = i + (1 - box_w) / 2
        dark = kind in ('hidden', 'output')
        ax.add_patch(FancyBboxPatch((x, box_y), box_w, box_h, boxstyle='round,pad=0.0,rounding_size=0.06',
                                    facecolor=FILLS[kind], edgecolor='none' if dark else EDGE, linewidth=1))
        ax.text(i + 0.5, box_y + box_h / 2, title, ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if dark else INK)
        ax.text(i + 0.5, box_y - 0.06, detail, ha='center', va='top', fontsize=8, color=MUTED, linespacing=1.4)
        if i < n - 1:
            ax.annotate('', xy=(i + 1 + (1 - box_w) / 2 - 0.02, box_y + box_h / 2), xytext=(x + box_w + 0.02, box_y + box_h / 2),
                        arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.4))
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved architecture diagram to {fig_path}')


if __name__ == '__main__':
    main()
