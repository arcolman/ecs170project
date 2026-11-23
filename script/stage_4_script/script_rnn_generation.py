"""Run from project root: python -m script.stage_4_script.script_rnn_generation [--prompt "what did the"]

Trains vanilla RNN, LSTM and GRU word-level joke generators with identical hyperparameters and one seed,
then one extra variant (2 stacked layers) of the model with the lowest final train loss.
Writes to result/stage_4_result/generation/:
  gen_loss_comparison.png   train loss per epoch, all models on one axis
  stories.txt               ~200 generated tokens per model from the same 3-word prompt
  results.md                final loss, train time, 5-gram copy rate, 3-gram repetition rate, notes
  history.json              raw per-epoch losses and metrics
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation, tokenize, detokenize, UNK, EOS
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation, pick_device

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'stage_4_data' / 'text_generation' / 'data'
OUT = ROOT / 'result' / 'stage_4_result' / 'generation'

SEED = 1
HP = dict(emb_dim=64, hidden_dim=128, num_layers=1, dropout=0.0)
TRAIN = dict(epochs=30, batch_size=32, lr=3e-3)
N_TOKENS = 200
TEMPERATURE = 0.8


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def story_metrics(story_tokens, train_5grams):
    g5, g3 = ngrams(story_tokens, 5), ngrams(story_tokens, 3)
    copy = sum(g in train_5grams for g in g5) / len(g5)
    rep = (len(g3) - len(set(g3))) / len(g3)
    return copy, rep


def run(name, cell, hp, data, prompt_ids, train_5grams, device):
    torch.manual_seed(SEED)
    model = Method_RNN_Generation(len(data['vocab']), cell=cell, **hp)
    print(f'Training {name} on {device}')
    history, secs = model.fit(data['jokes'], data['stoi'], device, seed=SEED, **TRAIN)
    gen = model.generate([data['stoi'][EOS]] + prompt_ids, N_TOKENS, device, temperature=TEMPERATURE, seed=SEED)
    tokens = [data['vocab'][i] for i in prompt_ids + gen]
    copy, rep = story_metrics(tokens, train_5grams)
    print(f'  {name}: final loss {history[-1]:.4f}, {secs:.1f}s, copy {copy:.1%}, repetition {rep:.1%}')
    return dict(name=name, cell=cell, hp=hp, history=history, seconds=secs,
                story=detokenize(tokens), copy_rate=copy, repetition_rate=rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='what did the', help='three starting words')
    ap.add_argument('--no-extra', action='store_true', help='skip the extra stacked-layer variant')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    data = Dataset_Loader_Generation(DATA).load()
    print(f'{len(data["jokes"])} jokes, vocab {len(data["vocab"])}, device {device}')

    prompt = tokenize(args.prompt)[:3]
    prompt_ids = [data['stoi'].get(t, data['stoi'][UNK]) for t in prompt]
    train_stream = [t for j in data['jokes'] for t in j[1:]]
    train_5grams = set(ngrams(train_stream, 5))

    runs = [run(n, c, HP, data, prompt_ids, train_5grams, device)
            for n, c in [('RNN', 'rnn'), ('LSTM', 'lstm'), ('GRU', 'gru')]]
    if not args.no_extra:
        best = min(runs, key=lambda r: r['history'][-1])
        extra_hp = dict(HP, num_layers=2)
        runs.append(run(f'{best["name"]} 2-layer', best['cell'], extra_hp, data, prompt_ids, train_5grams, device))

    # loss figure
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    for r in runs:
        ax.plot(range(1, len(r['history']) + 1), r['history'], label=r['name'], linewidth=1.4,
                linestyle='--' if 'layer' in r['name'] else '-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss (cross-entropy)')
    ax.set_title('Joke generator training loss', fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'gen_loss_comparison.png')

    with open(OUT / 'stories.txt', 'w', encoding='utf-8') as f:
        f.write(f'Prompt: "{" ".join(prompt)}", {N_TOKENS} sampled tokens, temperature {TEMPERATURE}, seed {SEED}.\n')
        f.write('Each new line is a generated <eos> (end of joke); the model keeps generating after it.\n\n')
        for r in runs:
            f.write(f'===== {r["name"]} =====\n{r["story"]}\n\n')

    with open(OUT / 'history.json', 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'story'} for r in runs], f, indent=1)

    with open(OUT / 'results.md', 'w', encoding='utf-8') as f:
        f.write('| Model | Final train loss | Train time (s) | 5-gram copy rate | 3-gram repetition rate | Coherence / grammar |\n')
        f.write('|---|---|---|---|---|---|\n')
        for r in runs:
            f.write(f'| {r["name"]} | {r["history"][-1]:.3f} | {r["seconds"]:.1f} | {r["copy_rate"]:.1%} '
                    f'| {r["repetition_rate"]:.1%} | TODO |\n')
        f.write(f'\nShared settings: word-level, embedding {HP["emb_dim"]}, hidden {HP["hidden_dim"]}, '
                f'{TRAIN["epochs"]} epochs, batch {TRAIN["batch_size"]}, Adam lr {TRAIN["lr"]}, grad clip 5, '
                f'seed {SEED}, device {device}. Prompt "{" ".join(prompt)}", {N_TOKENS} tokens, '
                f'temperature {TEMPERATURE}.\n')
    print(f'Wrote outputs to {OUT}')


if __name__ == '__main__':
    main()
