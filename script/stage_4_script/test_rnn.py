"""Run from project root: python -m script.stage_4_script.test_rnn

Checks the Stage 4 text cleaning, the IMDB loader and Method_RNN_Classification on small synthetic data, so it does not
need the datasets.
"""
from pathlib import Path
import tempfile
import numpy as np
import torch
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader_IMDB, tokenize, build_vocab, encode
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification


def check_text_processing():
    assert tokenize("This movie wasn't good.<br /><br />I can't recommend it!") == ['movie', 'not', 'good', 'not', 'recommend']
    assert tokenize('The END', remove_stopwords=False) == ['the', 'end']
    vocab = build_vocab([['b', 'a', 'b'], ['c', 'a']], min_freq=2)
    assert vocab == ['<pad>', '<unk>', 'a', 'b']
    X, lengths = encode([['a', 'zzz', 'b', 'a'], []], {w: i for i, w in enumerate(vocab)}, max_len=3)
    assert X.tolist() == [[2, 1, 3], [1, 0, 0]] and lengths.tolist() == [3, 1]


def check_loader():
    words = {'neg': 'awful boring bad', 'pos': 'great fun good'}
    with tempfile.TemporaryDirectory() as folder:
        # same layout as the instructor files: <split>/<pos|neg>/<id>_<rating>.txt
        for split in ('train', 'test'):
            for label, text in words.items():
                (Path(folder) / split / label).mkdir(parents=True)
                for i in range(10):
                    (Path(folder) / split / label / f'{i}_{8 if label == "pos" else 2}.txt').write_text(f'The film was {text}.<br />')
        loader = Dataset_Loader_IMDB('', '')
        loader.dataset_source_folder_path = folder
        loader.max_len, loader.min_freq, loader.val_fraction = 5, 1, 0.2
        data = loader.load()
        assert len(data['train']['y']) == 16 and np.bincount(data['val']['y']).tolist() == [2, 2]
        assert len(data['test']['y']) == 20 and data['train']['X'].shape == (16, 5)
        assert data['train']['lengths'].max() == 4  # 'film' + 3 sentiment words, stop words removed
        assert data['vocab'][:2] == ['<pad>', '<unk>']


def synthetic_reviews(n, seed, vocab_size=50, max_len=30):
    # label 1 reviews contain word 3 at three random places among random words, label 0 reviews word 4; lengths vary
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    lengths = rng.integers(5, max_len + 1, size=n)
    X = np.zeros((n, max_len), dtype=np.int64)
    for i in range(n):
        X[i, :lengths[i]] = rng.integers(5, vocab_size, size=lengths[i])
        X[i, rng.integers(0, lengths[i], size=3)] = 3 if y[i] else 4
    return {'X': X, 'lengths': lengths, 'y': y.astype(np.int64)}


def check_rnn():
    x = torch.randint(1, 50, (4, 12))
    lengths = torch.tensor([12, 3, 7, 1])
    # output portal: 2 logits per review, for every unit and architecture variant
    for settings in ({}, {'rnn_type': 'lstm'}, {'rnn_type': 'gru'}, {'rnn_type': 'lstm', 'bidirectional': True, 'num_layers': 2, 'dropout': 0.3},
                     {'pooling': 'mean'}, {'rnn_type': 'gru', 'pooling': 'max', 'bidirectional': True}):
        model = Method_RNN_Classification('rnn', '', vocab_size=50, embedding_dim=8, hidden_dim=16, **settings)
        assert model.forward(x, lengths).shape == (4, 2), settings
        # padding must not change the result: the same reviews with extra padding give the same logits
        # (model.eval() would call the template's train(), which shadows nn.Module.train)
        torch.nn.Module.train(model, False)
        padded = torch.cat([x, torch.zeros(4, 5, dtype=torch.long)], dim=1)
        assert torch.allclose(model.forward(x, lengths), model.forward(padded, lengths), atol=1e-5), settings

    weights = np.random.default_rng(0).normal(size=(50, 6)).astype(np.float32)
    model = Method_RNN_Classification('rnn', '', embedding_weights=weights, freeze_embedding=True)
    assert model.vocab_size == 50 and model.embedding_dim == 6 and not model.embedding.weight.requires_grad

    data = {'train': synthetic_reviews(1000, 1), 'val': synthetic_reviews(200, 2), 'test': synthetic_reviews(300, 3)}
    for rnn_type in ('rnn', 'lstm', 'gru'):
        model = Method_RNN_Classification(rnn_type, '', rnn_type=rnn_type, vocab_size=50, embedding_dim=16, hidden_dim=32, max_epoch=8)
        model.data = data
        result = model.run()
        accuracy = (result['pred_y'].numpy() == data['test']['y']).mean()
        print(f'{rnn_type}: loss {model.history["train_loss"][0]:.3f} -> {model.history["train_loss"][-1]:.3f}, '
              f'test accuracy {accuracy:.3f}')
        assert model.history['train_loss'][-1] < model.history['train_loss'][0]
        assert len(model.history['val_accuracy']) == 8
        assert accuracy > 0.9, rnn_type

    for bad in ({'rnn_type': 'transformer'}, {'hiden_dim': 3}, {'pooling': 'sum'}):
        try:
            Method_RNN_Classification('rnn', '', vocab_size=50, **bad)
            raise AssertionError(f'bad setting was accepted: {bad}')
        except ValueError:
            pass


def main():
    check_text_processing()
    check_loader()
    check_rnn()
    print('Stage 4 checks passed.')


if __name__ == '__main__':
    main()
