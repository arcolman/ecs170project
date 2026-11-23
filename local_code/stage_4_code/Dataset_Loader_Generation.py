"""Word-level loader for the Stage 4 text generation dataset (1622 short jokes, CSV with columns ID, Joke).

Each joke is lowercased and split into words and punctuation, and wrapped as <eos> w1 ... wn <eos>.
The leading <eos> acts as a start token, so the model learns how a joke begins after the previous one ends.
"""
import csv
import re
from pathlib import Path

PAD, EOS, UNK = '<pad>', '<eos>', '<unk>'
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*|[^\sa-z0-9]")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens):
    out = ''
    for t in tokens:
        if t == EOS:
            out += '\n'
        elif out == '' or out.endswith('\n') or not re.match(r'[a-z0-9]', t):
            out += t
        else:
            out += ' ' + t
    return out.replace('\n ', '\n')


class Dataset_Loader_Generation:
    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load(self):
        with open(self.data_path, encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        jokes = [[EOS] + tokenize(r['Joke']) + [EOS] for r in rows if r['Joke'].strip()]
        vocab = [PAD, EOS, UNK] + sorted({t for j in jokes for t in j} - {EOS})
        stoi = {w: i for i, w in enumerate(vocab)}
        return {'jokes': jokes, 'vocab': vocab, 'stoi': stoi}
