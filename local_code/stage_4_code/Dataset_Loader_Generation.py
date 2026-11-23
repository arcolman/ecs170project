"""Word-level joke data. Keep punctuation/stop words; split before vocabulary fitting."""
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from local_code.base_class.dataset import dataset

SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN.findall(text.lower().replace('\u2019', "'"))


def detokenize(tokens):
    text = ' '.join(tokens)
    text = re.sub(r'\s+([.,!?;:%)\]])', r'\1', text)
    return re.sub(r'([([])\s+', r'\1', text)


class Dataset_Loader_Generation(dataset):
    def __init__(self, name='jokes', description='', source=None, split_seed=42,
                 max_vocab_size=20000, max_tokens=128):
        super().__init__(name, description)
        self.source = Path(source) if source else None
        self.split_seed = split_seed
        self.max_vocab_size = max_vocab_size
        self.max_tokens = max_tokens

    def load(self):
        with self.source.open(encoding='utf-8-sig', newline='') as f:
            rows = [{'id': r['ID'], 'text': r['Joke'], 'tokens': tokenize(r['Joke'])}
                    for r in csv.DictReader(f) if r['Joke'].strip()]
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row['tokens'])].append(row)
        keys = sorted(groups)
        random.Random(self.split_seed).shuffle(keys)
        nval = max(1, round(len(keys) * .1))
        val_keys = set(keys[:nval])
        train = [r for r in rows if tuple(r['tokens']) not in val_keys]
        val = [r for r in rows if tuple(r['tokens']) in val_keys]
        counts = Counter(t for r in train for t in r['tokens'][:self.max_tokens])
        vocab = SPECIAL + sorted(counts, key=lambda t: (-counts[t], t))[:self.max_vocab_size-len(SPECIAL)]
        lookup = {t: i for i, t in enumerate(vocab)}
        def encode(part):
            return [[lookup.get(t, 1) for t in r['tokens'][:self.max_tokens]] for r in part]
        lengths = [len(r['tokens']) for r in rows]
        return {
            'train': encode(train), 'val': encode(val), 'vocab': vocab,
            'train_rows': train, 'val_rows': val,
            'stats': {'samples': len(rows), 'unique_token_sequences': len(keys),
                      'train_samples': len(train), 'validation_samples': len(val),
                      'split_seed': self.split_seed, 'vocabulary_size': len(vocab),
                      'mean_tokens': sum(lengths)/len(lengths), 'max_tokens_observed': max(lengths),
                      'sequence_token_limit': self.max_tokens,
                      'truncated_samples': sum(n > self.max_tokens for n in lengths),
                      'validation_unknown_fraction': sum(t not in lookup for r in val for t in r['tokens']) /
                          max(1, sum(len(r['tokens']) for r in val))},
        }
