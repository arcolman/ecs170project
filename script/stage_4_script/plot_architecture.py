"""Run from project root: python -m script.stage_4_script.plot_architecture [rnn|lstm|gru]

Draws the recurrent classifier for the report (Section 2) to result/stage_4_result/classification/cls_<unit>_architecture.png:
(a) the layer pipeline with the baseline sizes, (b) the recurrent layer unrolled over one example review, showing that the
hidden state is carried word to word and that padding is skipped. Sizes are read from Method_RNN_Classification's defaults
and the loader's settings, so the figure matches the trained baseline.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification as M
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader_IMDB as L, tokenize

INK, MUTED, EDGE = '#1f1f1d', '#6b6a66', '#8a8984'
FILLS = {'input': '#e8f1fb', 'embed': '#7fb0ea', 'hidden': '#2a78d6', 'output': '#eb6834', 'loss': '#f4f3f0'}
UNITS = {'rnn': ('RNN', 'tanh'), 'lstm': ('LSTM', 'gated, cell state'), 'gru': ('GRU', 'gated')}
EXAMPLE = "This movie wasn't good at all."


def box(ax, x, y, w, h, kind, text, fontsize=10, dashed=False):
    dark = kind in ('hidden', 'output', 'embed')
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.05',
                                facecolor='white' if dashed else FILLS[kind], linewidth=1,
                                edgecolor=EDGE if not dark or dashed else 'none', linestyle='--' if dashed else '-'))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold',
            color=MUTED if dashed else ('white' if dark else INK))


def arrow(ax, start, end, color=EDGE, style='-|>'):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle=style, color=color, lw=1.4, shrinkA=0, shrinkB=0))


def main():
    unit = sys.argv[1] if len(sys.argv) > 1 else 'rnn'
    unit_name, unit_detail = UNITS[unit]
    root = Path(__file__).resolve().parents[2]
    fig_path = root / 'result' / 'stage_4_result' / 'classification' / f'cls_{unit}_architecture.png'
    E, H, V, T = M.embedding_dim, M.hidden_dim, L.max_vocab_size, L.max_len

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 6.3)
    ax.set_ylim(0, 4.6)
    ax.axis('off')

    # (a) pipeline
    ax.text(0.02, 4.5, '(a) Layers', fontsize=10, fontweight='bold', color=INK, va='top')
    stages = [('input', 'Review', f'word ids, padded/cut\nto {T} tokens'),
              ('embed', 'Embedding', f'{V:,} words → {E}-d\n(trained from scratch)'),
              ('hidden', f'{unit_name}\n{H} units', f'{E} → {H}, {unit_detail}\n1 layer, packed sequence'),
              ('hidden', f'$h_T$\n{H}-d', 'hidden state after\nlast real word'),
              ('output', 'Output\n2 logits', f'Linear {H} → 2'),
              ('loss', 'Softmax', 'train: cross-entropy\ntest: argmax = neg/pos')]
    bw, bh, by = 0.78, 0.72, 3.35
    for i, (kind, title, detail) in enumerate(stages):
        x = i + (1 - bw) / 2
        box(ax, x, by, bw, bh, kind, title)
        ax.text(i + 0.5, by - 0.08, detail, ha='center', va='top', fontsize=8, color=MUTED, linespacing=1.4)
        if i < len(stages) - 1:
            arrow(ax, (x + bw + 0.02, by + bh / 2), (i + 1 + (1 - bw) / 2 - 0.02, by + bh / 2))

    # (b) unrolled over an example review: tokens -> embeddings -> hidden states, padding skipped
    tokens = tokenize(EXAMPLE)
    ax.text(0.02, 2.3, f'(b) {unit_name} unrolled over one review', fontsize=10, fontweight='bold', color=INK, va='top')
    ax.text(0.02, 2.02, f'"{EXAMPLE}"\n→ cleaned: {" ".join(tokens)}', fontsize=8, color=MUTED, va='top', linespacing=1.5)
    cols = tokens + ['<pad>', '<pad>']
    x0, step, cw = 1.55, 0.72, 0.5
    y_word, y_emb, y_h = 0.12, 0.5, 1.32
    for t, word in enumerate(cols):
        cx = x0 + t * step
        pad = word == '<pad>'
        ax.text(cx + cw / 2, y_word + 0.12, word, ha='center', va='center', fontsize=9, color=MUTED if pad else INK,
                style='italic' if pad else 'normal')
        box(ax, cx, y_emb, cw, 0.36, 'embed', f'$x_{t + 1}$', fontsize=8, dashed=pad)
        box(ax, cx, y_h, cw, 0.5, 'hidden', f'$h_{t + 1}$', fontsize=9, dashed=pad)
        if not pad:
            arrow(ax, (cx + cw / 2, y_word + 0.26), (cx + cw / 2, y_emb - 0.02))
            arrow(ax, (cx + cw / 2, y_emb + 0.38), (cx + cw / 2, y_h - 0.02))
        if t < len(cols) - 1 and not pad and cols[t + 1] != '<pad>':
            arrow(ax, (cx + cw + 0.02, y_h + 0.25), (cx + step - 0.02, y_h + 0.25))
    # h0 and the read-out from the last real word
    ax.text(x0 - 0.2, y_h + 0.25, '$h_0 = 0$', ha='right', va='center', fontsize=8, color=MUTED)
    arrow(ax, (x0 - 0.18, y_h + 0.25), (x0 - 0.02, y_h + 0.25))
    last = x0 + (len(tokens) - 1) * step
    pad_mid = x0 + (len(tokens) + 0.5) * step + cw / 2
    ax.text(pad_mid, (y_emb + 0.36 + y_h) / 2, 'padding skipped (packed)', ha='center', va='center', fontsize=8, color=MUTED)
    out_x = x0 + len(cols) * step + 0.15
    ax.annotate('', xy=(out_x - 0.02, y_h + 0.95), xytext=(last + cw / 2, y_h + 0.52),
                arrowprops=dict(arrowstyle='-|>', color=FILLS['output'], lw=1.6, connectionstyle='angle,angleA=90,angleB=180,rad=8'))
    box(ax, out_x, y_h + 0.72, 0.85, 0.46, 'output', 'Linear → 2', fontsize=9)
    ax.text(out_x + 0.95, y_h + 0.95, 'logits → softmax:\nP(negative), P(positive)', ha='left', va='center', fontsize=8, color=MUTED, linespacing=1.4)
    rule = {'rnn': r'$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$', 'lstm': 'gates decide what the cell state keeps, adds and outputs',
            'gru': r'update / reset gates mix $h_{t-1}$ with a new candidate state'}[unit]
    ax.text(out_x, y_h + 0.25, f'same weights at every step:\n{rule}', ha='left', va='center', fontsize=8, color=MUTED, linespacing=1.5)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved architecture diagram to {fig_path}')


if __name__ == '__main__':
    main()
