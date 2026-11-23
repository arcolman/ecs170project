"""Checks generation preprocessing, causal logits, reproducibility and overlap metrics."""
import csv
import tempfile
import unittest
from pathlib import Path
import torch
from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation, tokenize
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Generation_Helper import generate, compare_to_training
from script.stage_4_script.script_rnn_gen import collate


class GenerationTests(unittest.TestCase):
    def test_split_and_vocabulary(self):
        # Keep temporary files within the task's existing work directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as temp:
            path = Path(temp) / 'jokes.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Joke'])
                for i in range(30):
                    writer.writerow([i, f'What did the animal {i} do? It did not sleep!'])
                writer.writerow([30, 'WHAT DID THE ANIMAL 1 DO? IT DID NOT SLEEP!'])
            data = Dataset_Loader_Generation(source=path).load()
            train_keys = {tuple(r['tokens']) for r in data['train_rows']}
            val_keys = {tuple(r['tokens']) for r in data['val_rows']}
            self.assertFalse(train_keys & val_keys)
            train_words = {w for r in data['train_rows'] for w in r['tokens']}
            self.assertEqual(set(data['vocab'][4:]), train_words)
            self.assertEqual(len(data['train']) + len(data['val']), 31)
        self.assertIn('not', tokenize('It did not sleep!'))
        self.assertIn('!', tokenize('It did not sleep!'))

    def test_shift_and_padding(self):
        x, y = collate([[4, 5], [6]])
        self.assertEqual(x.tolist(), [[2, 4, 5], [2, 6, 0]])
        self.assertEqual(y.tolist(), [[4, 5, 3], [6, 3, 0]])

    def test_causality_and_sampling(self):
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>', 'what', 'did', 'the', 'cat', '?']
        torch.set_num_threads(2)
        for unit in ('rnn', 'lstm', 'gru'):
            torch.manual_seed(1)
            model = Method_RNN_Generation(len(vocab), embedding_dim=8, hidden_dim=12, rnn_type=unit)
            model.eval()
            a, _ = model(torch.tensor([[2, 4, 5, 6]]))
            b, _ = model(torch.tensor([[2, 4, 7, 8]]))
            torch.testing.assert_close(a[:, :2], b[:, :2])
            first = generate(model, vocab, seed=7, max_new_tokens=10)
            self.assertEqual(first, generate(model, vocab, seed=7, max_new_tokens=10))
            self.assertEqual(first['tokens'][:3], ['what', 'did', 'the'])
            self.assertFalse(set(first['tokens']) & set(vocab[:4]))

    def test_overlap_excludes_prompt(self):
        sample = {'tokens': tokenize('what did the cat eat the small fish')}
        rows = [{'id': '1', 'text': 'cat eat the small fish', 'tokens': tokenize('cat eat the small fish')}]
        result = compare_to_training(sample, rows)
        self.assertEqual(result['longest_training_match_words'], 5)
        self.assertEqual(result['training_ngram_match_fraction'], 1.)
        short = compare_to_training({'tokens': tokenize('what did the')}, rows)
        self.assertIsNone(short['training_ngram_match_fraction'])
        self.assertEqual(short['longest_training_match_words'], 0)


if __name__ == '__main__':
    unittest.main()
