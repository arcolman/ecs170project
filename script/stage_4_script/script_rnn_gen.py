"""Train/generate jokes: python -m script.stage_4_script.script_rnn_gen [--unit rnn|lstm|gru]."""
import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Generation_Helper import generate, compare_to_training


def collate(rows):
    inputs = [torch.tensor([2] + r, dtype=torch.long) for r in rows]
    targets = [torch.tensor(r + [3], dtype=torch.long) for r in rows]
    return pad_sequence(inputs, batch_first=True), pad_sequence(targets, batch_first=True)


def epoch_pass(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            ntokens = int(y.ne(0).sum())
            if optimizer is not None:
                optimizer.zero_grad()
                (loss/ntokens).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            total += float(loss.detach())
            count += ntokens
    return total/count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', choices=['rnn', 'lstm', 'gru'], default='rnn')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--run-name')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    root = Path(__file__).resolve().parents[2]
    folder = root / 'result/stage_4_result/generation'
    folder.mkdir(parents=True, exist_ok=True)
    name = args.run_name or args.unit
    summary_path = folder / f'gen_{name}_summary.json'
    if summary_path.exists() and not args.fresh:
        print(f'Run {name} is complete; use --fresh to rerun it.')
        return
    data = Dataset_Loader_Generation(source=root/'data/stage_4_data/text_generation/data').load()
    print(json.dumps(data['stats'], indent=2), flush=True)
    (folder/'generation_data_stats.json').write_text(json.dumps(data['stats'], indent=2))
    (folder/'generation_split.json').write_text(json.dumps({
        'split_seed': 42, 'train_ids': [r['id'] for r in data['train_rows']],
        'validation_ids': [r['id'] for r in data['val_rows']]}, indent=2))
    (folder/'generation_vocabulary.json').write_text(json.dumps(data['vocab']))
    rng = torch.Generator().manual_seed(args.seed)
    train = DataLoader(data['train'], batch_size=64, shuffle=True, generator=rng, collate_fn=collate)
    val = DataLoader(data['val'], batch_size=64, collate_fn=collate)
    model = Method_RNN_Generation(len(data['vocab']), hidden_dim=args.hidden_dim,
                                  num_layers=args.layers, rnn_type=args.unit, dropout=args.dropout)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best_loss, best_epoch, best_state = float('inf'), 0, None
    start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        train_loss = epoch_pass(model, train, device, optimizer)
        val_loss = epoch_pass(model, val, device)
        for key, value in zip(history, (epoch, train_loss, val_loss)):
            history[key].append(value)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f'{name} epoch {epoch}/{args.epochs}: train={train_loss:.4f} val={val_loss:.4f} '
              f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
        (folder/f'gen_{name}_history.json').write_text(json.dumps(history, indent=2))
    duration = time.perf_counter()-start
    model.load_state_dict(best_state)
    # Prefixes occur in the provided data. All units reuse the same prompts and sampling settings.
    prompts = ['what did the', 'why did the', 'how do you']
    samples = []
    for prompt in prompts:
        for sample_seed in (1, 2, 3):
            sample = generate(model, data['vocab'], prompt, seed=sample_seed)
            sample['comparison'] = compare_to_training(sample, data['train_rows'])
            samples.append(sample)
    torch.save({'config': model.config, 'state_dict': model.state_dict(),
                'vocab': data['vocab'], 'best_epoch': best_epoch, 'args': vars(args)},
               folder/f'gen_{name}_checkpoint.pt')
    (folder/f'gen_{name}_samples.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.plot(history['epoch'], history['train_loss'], '-o', label='Training', markersize=4)
    ax.plot(history['epoch'], history['val_loss'], '--s', label='Validation', markersize=4)
    ax.axvline(best_epoch, color='#777777', alpha=.5, linestyle=':', label='Selected epoch')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy per target token', title=f'Joke generation: {name}')
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder/f'gen_{name}_loss.png', dpi=200)
    plt.close(fig)
    summary = {'name': name, 'unit': args.unit, 'model_config': model.config,
               'seed': args.seed, 'split_seed': 42, 'epochs': args.epochs,
               'batch_size': 64, 'optimizer': 'Adam', 'learning_rate': .001,
               'gradient_clip_norm': 1., 'device': str(device), 'threads': 4,
               'best_epoch': best_epoch, 'validation_loss': best_loss,
               'validation_perplexity': math.exp(best_loss), 'training_seconds': duration,
               'parameters': sum(p.numel() for p in model.parameters()), 'torch_version': torch.__version__,
               'data': data['stats'], 'prompts': prompts, 'sample_seeds': [1, 2, 3],
               'temperature': .8, 'top_k': 20, 'max_new_tokens': 60}
    summary_path.write_text(json.dumps(summary, indent=2))
    summaries = [json.loads(p.read_text()) for p in sorted(folder.glob('gen_*_summary.json'))]
    with (folder/'generation_results.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['run', 'epochs', 'best_epoch', 'validation_loss', 'validation_perplexity', 'training_seconds'])
        for s in summaries:
            writer.writerow([s['name'], s['epochs'], s['best_epoch'], s['validation_loss'],
                             s['validation_perplexity'], s['training_seconds']])
    print(json.dumps(summary, indent=2), flush=True)
    for s in samples:
        print(s['prompt'], 'seed', s['seed'], ':', s['text'], flush=True)


if __name__ == '__main__':
    main()
