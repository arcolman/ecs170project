"""Load the instructor-provided Stage 4 text classification dataset (IMDB movie reviews, positive vs negative).

Folder layout in data/stage_4_data/text_classification/: {train, test}/{pos, neg}/<id>_<rating>.txt, one review per file,
25,000 train / 25,000 test, balanced. The given train/test split is kept as is; a fixed, stratified slice of the training
set is held out as a validation set for the per-epoch learning curves (the test set is only used once, at the end).

Pipeline: clean_text -> tokenize -> vocabulary built from the training split only -> word ids, padded / truncated to
max_len. load() returns {'train', 'val', 'test'}, each {'X', 'lengths', 'y'}:
  X        int64 (N, max_len) word ids, 0 = <pad> (after the text), 1 = <unk> (word not in the vocabulary)
  lengths  int64 (N,) number of real tokens before the padding (at least 1), used to pack the sequences for the RNN
  y        int64 (N,) 0 = negative, 1 = positive
plus 'vocab' (list of words, index = word id) and 'raw' (the cleaned token lists, for data checks).
"""
from pathlib import Path
from collections import Counter
import re
import numpy as np
from local_code.base_class.dataset import dataset

PAD, UNK = '<pad>', '<unk>'
LABELS = {'neg': 0, 'pos': 1}

# common English stop words (based on NLTK's list), minus the negations and intensifiers that flip or scale sentiment:
# "not good" and "good" must not become the same text
STOP_WORDS = frozenset('''
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just let me more most my myself of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up us was we were what when where which while who whom why will with would you
your yours yourself yourselves s t d ll m o re ve y
'''.split())
# kept on purpose: no, nor, not, never, very, against... (and the n't contractions, see clean_text)

HTML_TAG = re.compile(r'<[^>]+>')
NON_LETTER = re.compile(r"[^a-z']+")


def clean_text(text):
    '''Raw review -> lowercase string of words. Removes HTML line breaks (<br />), punctuation and digits.'''
    text = HTML_TAG.sub(' ', text.lower())
    # keep the negation in contractions as a separate word: "didn't" -> "did not"
    text = text.replace("can't", 'can not').replace("won't", 'will not').replace("n't", ' not')
    text = NON_LETTER.sub(' ', text)
    return text.replace("'", ' ')


def tokenize(text, remove_stopwords=True):
    '''Raw review -> list of tokens.'''
    tokens = clean_text(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def build_vocab(token_lists, max_vocab_size=None, min_freq=1):
    '''Most frequent words first; ids 0 and 1 are reserved for <pad> and <unk>. Ties are broken alphabetically, so the
    vocabulary (and therefore every word id) is the same on every machine.'''
    counts = Counter(t for tokens in token_lists for t in tokens)
    words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
    if max_vocab_size is not None:
        words = words[:max_vocab_size - 2]
    return [PAD, UNK] + words


def encode(token_lists, word_to_id, max_len):
    '''Token lists -> (X, lengths): word ids padded with 0 after the text, texts longer than max_len cut at the end.'''
    X = np.zeros((len(token_lists), max_len), dtype=np.int64)
    lengths = np.zeros(len(token_lists), dtype=np.int64)
    unk = word_to_id[UNK]
    for i, tokens in enumerate(token_lists):
        ids = [word_to_id.get(t, unk) for t in tokens[:max_len]] or [unk]  # an empty review becomes one <unk>
        X[i, :len(ids)] = ids
        lengths[i] = len(ids)
    return X, lengths


def load_glove(path, vocab, dim, seed=0):
    '''Embedding matrix (len(vocab), dim) from a GloVe text file (e.g. glove.6B.100d.txt): rows of words found in GloVe get
    their GloVe vector, the others small random values, <pad> stays zero. Pass it as Method_RNN_Classification's
    embedding_weights setting.'''
    word_to_id = {w: i for i, w in enumerate(vocab)}
    weights = np.random.default_rng(seed).normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    weights[0] = 0
    found = 0
    with open(path, encoding='utf8') as f:
        for line in f:
            word, *values = line.rstrip().split(' ')
            if word in word_to_id and len(values) == dim:
                weights[word_to_id[word]] = np.asarray(values, dtype=np.float32)
                found += 1
    print(f'GloVe: found {found} of {len(vocab)} vocabulary words')
    return weights


class Dataset_Loader_IMDB(dataset):
    dataset_source_folder_path = None  # the text_classification folder
    max_len = 200            # tokens kept per review (about the 85th percentile of the cleaned lengths)
    max_vocab_size = 20000   # most frequent training words, the rest become <unk>
    min_freq = 2             # a word must appear at least this often in the training set
    remove_stopwords = True
    val_fraction = 0.1       # share of the training set held out for validation (0 = none)
    seed = 42                # fixes the validation split, agreed by the whole team
    num_classes = 2

    def read_split(self, split):
        '''-> (texts, labels, ratings) for 'train' or 'test', files in sorted order so every machine sees the same order.'''
        texts, labels, ratings = [], [], []
        for label_name, label in LABELS.items():
            folder = Path(self.dataset_source_folder_path) / split / label_name
            files = sorted(folder.glob('*.txt'))
            if not files:
                raise FileNotFoundError(f'no reviews found in {folder}')
            for file in files:
                texts.append(file.read_text(encoding='utf8'))
                labels.append(label)
                ratings.append(int(file.stem.split('_')[1]))  # file name is <id>_<rating>.txt
        return texts, np.array(labels, dtype=np.int64), np.array(ratings, dtype=np.int64)

    def validation_indices(self, y):
        '''Stratified random slice of the training set: same size share of each class, fixed by the seed.'''
        rng = np.random.default_rng(self.seed)
        val = []
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            val.extend(rng.choice(idx, size=int(round(len(idx) * self.val_fraction)), replace=False))
        val = np.sort(np.array(val, dtype=np.int64))
        return np.setdiff1d(np.arange(len(y)), val), val

    def load(self):
        print(f'loading data from {self.dataset_source_folder_path}...')
        tokens, labels = {}, {}
        for split in ('train', 'test'):
            texts, labels[split], _ = self.read_split(split)
            tokens[split] = [tokenize(t, self.remove_stopwords) for t in texts]

        # split off validation before building the vocabulary, so validation words count as unseen like test words
        train_idx, val_idx = self.validation_indices(labels['train']) if self.val_fraction > 0 else (np.arange(len(labels['train'])), None)
        splits = {'train': ([tokens['train'][i] for i in train_idx], labels['train'][train_idx])}
        if val_idx is not None:
            splits['val'] = ([tokens['train'][i] for i in val_idx], labels['train'][val_idx])
        splits['test'] = (tokens['test'], labels['test'])

        vocab = build_vocab(splits['train'][0], self.max_vocab_size, self.min_freq)
        word_to_id = {w: i for i, w in enumerate(vocab)}
        data = {'vocab': vocab, 'raw': {}}
        for split, (token_lists, y) in splits.items():
            X, lengths = encode(token_lists, word_to_id, self.max_len)
            data[split] = {'X': X, 'lengths': lengths, 'y': y}
            data['raw'][split] = token_lists
            print(f'{split}: X={X.shape}, {np.bincount(y).tolist()} neg/pos, '
                  f'{(lengths == self.max_len).mean():.1%} truncated to {self.max_len} tokens')
        print(f'vocabulary: {len(vocab)} words (incl. <pad>, <unk>)')
        return data
