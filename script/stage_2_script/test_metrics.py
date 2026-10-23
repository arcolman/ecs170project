"""Run from project root: python -m script.stage_2_script.test_metrics

Checks Evaluate_Metrics on synthetic predictions, so it does not need the dataset or a trained MLP.
"""
import numpy as np
import torch
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics


def check(name, true_y, pred_y):
    evaluator = Evaluate_Metrics('multiclass metrics', '')
    evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
    scores = evaluator.evaluate()
    print(f'--- {name}')
    for metric, value in scores.items():
        print(f'{metric:>20}: {value:.4f}')
    return scores


def main():
    rng = np.random.default_rng(0)
    true_y = rng.integers(0, 10, size=1000)

    # perfect predictions: every metric should be exactly 1
    scores = check('perfect', true_y, true_y.copy())
    assert all(v == 1.0 for v in scores.values())

    # ~70% correct: accuracy ~0.7, and micro-averaged P/R/F1 equal accuracy for single-label multiclass
    pred_y = true_y.copy()
    wrong = rng.random(1000) < 0.3
    pred_y[wrong] = rng.integers(0, 10, size=wrong.sum())
    scores = check('~70% correct (numpy)', true_y, pred_y)
    assert 0.6 < scores['accuracy'] < 0.8
    for metric in ('precision_micro', 'recall_micro', 'f1_micro'):
        assert np.isclose(scores[metric], scores['accuracy'])

    # same predictions as torch tensors, the way Method_MLP.test() returns them
    torch_scores = check('~70% correct (torch)', torch.from_numpy(true_y), torch.from_numpy(pred_y))
    assert torch_scores == scores

    print('Metric checks passed.')


if __name__ == '__main__':
    main()
