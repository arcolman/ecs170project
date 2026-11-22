"""Run from project root: python -m script.stage_4_script.check_data

Step 4-1 for the classification dataset: loads the IMDB reviews from data/stage_4_data/text_classification/ and prints
the numbers for the report's dataset section (sizes, class balance, star ratings, review lengths before and after
cleaning, vocabulary, unknown-word rate, example reviews). Saves the printout and a review-length histogram to
result/stage_4_result/data_check/.
"""
from pathlib import Path
import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader_IMDB, clean_text, tokenize


def describe_lengths(name, lengths):
    lengths = np.asarray(lengths)
    print(f'  {name}: mean {lengths.mean():.0f}, median {np.median(lengths):.0f}, 85th pct {np.percentile(lengths, 85):.0f}, '
          f'95th pct {np.percentile(lengths, 95):.0f}, min {lengths.min()}, max {lengths.max()}')


def main():
    root = Path(__file__).resolve().parents[2]
    out_folder = root / 'result' / 'stage_4_result' / 'data_check'
    out_folder.mkdir(parents=True, exist_ok=True)
    loader = Dataset_Loader_IMDB('IMDB', 'movie review sentiment')
    loader.dataset_source_folder_path = root / 'data' / 'stage_4_data' / 'text_classification'

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        raw = {}
        for split in ('train', 'test'):
            texts, y, ratings = loader.read_split(split)
            raw[split] = texts
            print(f'{split}: {len(texts)} reviews, {np.bincount(y).tolist()} neg/pos')
            print(f'  star ratings of negative reviews {sorted(set(ratings[y == 0].tolist()))}, positive {sorted(set(ratings[y == 1].tolist()))}')
            describe_lengths('raw words per review', [len(t.split()) for t in texts])
            describe_lengths('tokens after cleaning (stop words removed)', [len(tokenize(t)) for t in texts])
            print(f'  reviews containing HTML line breaks: {np.mean(["<br" in t for t in texts]):.1%}')

        print('\nexample reviews (train):')
        for i in (0, len(raw['train']) - 1):
            print(f'  raw:     {raw["train"][i][:300]}')
            print(f'  cleaned: {" ".join(tokenize(raw["train"][i]))[:300]}\n')

        print('loader output:')
        data = loader.load()
        vocab = data['vocab']
        all_train_words = {t for tokens in data['raw']['train'] for t in tokens}
        print(f'distinct words in the training split: {len(all_train_words)}, kept in the vocabulary: {len(vocab) - 2}')
        print(f'20 most frequent words: {vocab[2:22]}')
        for split in ('train', 'val', 'test'):
            d = data[split]
            mask = np.arange(d['X'].shape[1])[None, :] < d['lengths'][:, None]
            print(f'{split}: {(d["X"][mask] == 1).mean():.1%} of tokens are <unk>')
            assert d['X'].dtype == np.int64 and d['X'].shape[1] == loader.max_len
            assert d['lengths'].min() >= 1 and d['lengths'].max() <= loader.max_len
            assert (d['X'][~mask] == 0).all(), 'padding must come after the text'
            assert set(np.unique(d['y'])) == {0, 1}
        assert len(data['train']['y']) + len(data['val']['y']) == 25000 and len(data['test']['y']) == 25000
        print('Dataset checks passed. Ready for RNN training.')
    print(log.getvalue(), end='')
    (out_folder / 'check_data_output.txt').write_text(log.getvalue())

    lengths = [len(tokenize(t)) for t in raw['train']]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.hist(lengths, bins=80, range=(0, 1000), color='#2a78d6')
    ax.axvline(loader.max_len, color='#eb6834', linewidth=2, label=f'max_len = {loader.max_len}')
    ax.set_xlabel('Tokens per training review (after cleaning)')
    ax.set_ylabel('Reviews')
    ax.legend(frameon=False, fontsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_folder / 'cls_review_lengths.png', dpi=200)
    plt.close(fig)
    print(f'saved output and length histogram to {out_folder}')


if __name__ == '__main__':
    main()
