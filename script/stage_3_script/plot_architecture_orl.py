"""Run from project root: python -m script.stage_3_script.plot_architecture_orl [baseline|final]

Draws the ORL CNN architecture (orl_configs.FINAL by default) for the report (Section 2) to
result/stage_3_result/orl/<run>_architecture.png, with the feature-map size after every block measured from the model.
Same visual style as the Stage 2 MLP diagram.
"""
from pathlib import Path
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from script.stage_3_script.orl_configs import BASELINE, FINAL

INK, MUTED, EDGE = '#1f1f1d', '#6b6a66', '#8a8984'
FILLS = {'input': '#e8f1fb', 'conv': '#2a78d6', 'fc': '#1baf7a', 'output': '#eb6834', 'loss': '#f4f3f0'}


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else 'final'
    settings = {'baseline': BASELINE, 'final': FINAL}[run_name]
    root = Path(__file__).resolve().parents[2]
    fig_path = root / 'result' / 'stage_3_result' / 'orl' / f'{run_name}_architecture.png'
    model = Method_CNN_ORL('cnn', '', **settings)
    # the template train(X, y) shadows nn.Module.train, so model.eval() cannot be used
    torch.nn.Module.train(model, False)
    k, pool = model.kernel_size, {'max': 'max', 'avg': 'avg'}[model.pool]

    stages = [('input', 'Input', '1 x 112 x 92\ngrayscale face\n(scaled to [0,1])')]
    x = torch.zeros(1, 1, 112, 92)
    c_in = 1
    blocks = [m for m in model.features]
    # walk the feature extractor block by block to report each output size
    per_block = len(blocks) // len(model.conv_channels)
    for i, c in enumerate(model.conv_channels):
        with torch.no_grad():
            for layer in blocks[i * per_block:(i + 1) * per_block]:
                x = layer(x)
        bn = '\nBatchNorm' if model.batch_norm else ''
        stages.append(('conv', f'Conv block {i + 1}',
                       f'Conv {k}x{k}, {c_in}→{c}{bn}\nReLU, {pool}-pool 2x2\n→ {c} x {x.shape[2]} x {x.shape[3]}'))
        c_in = c
    flat = x.numel()
    prev = flat
    for j, h in enumerate(model.fc_dims, 1):
        drop = f'\nDropout p={model.dropout}' if model.dropout else ''
        stages.append(('fc', f'FC {j}\n{h} units', f'flatten {flat}{drop}\nLinear {prev}→{h}, ReLU'))
        prev = h
    drop = f'Dropout p={model.dropout}\n' if model.dropout else ''
    flat_text = f'flatten {flat}\n' if not model.fc_dims else ''
    stages.append(('output', 'Output\n40 logits', f'{flat_text}{drop}Linear {prev}→40'))
    stages.append(('loss', 'Softmax', 'train: cross-entropy\ntest: argmax =\nperson 1-40'))

    n = len(stages)
    fig, ax = plt.subplots(figsize=(1.6 * n, 2.2))
    ax.set_xlim(0, n)
    ax.set_ylim(0.12, 0.88)
    ax.axis('off')
    box_w, box_h, box_y = 0.8, 0.34, 0.5
    for i, (kind, title, detail) in enumerate(stages):
        x0 = i + (1 - box_w) / 2
        dark = kind in ('conv', 'fc', 'output')
        ax.add_patch(FancyBboxPatch((x0, box_y), box_w, box_h, boxstyle='round,pad=0.0,rounding_size=0.06',
                                    facecolor=FILLS[kind], edgecolor='none' if dark else EDGE, linewidth=1))
        ax.text(i + 0.5, box_y + box_h / 2, title, ha='center', va='center', fontsize=9.5, fontweight='bold',
                color='white' if dark else INK)
        ax.text(i + 0.5, box_y - 0.05, detail, ha='center', va='top', fontsize=7.5, color=MUTED, linespacing=1.4)
        if i < n - 1:
            ax.annotate('', xy=(i + 1 + (1 - box_w) / 2 - 0.02, box_y + box_h / 2),
                        xytext=(x0 + box_w + 0.02, box_y + box_h / 2),
                        arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.4))
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved architecture diagram to {fig_path} '
          f'({sum(p.numel() for p in model.parameters()):,} trainable parameters)')


if __name__ == '__main__':
    main()
