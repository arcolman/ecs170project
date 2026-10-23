"""Run from project root: python -m script.stage_2_script.test_result_report

Builds the results log, table, and comparison plot from fake predictions, so it does not need the dataset or a trained MLP.
Output goes to result/stage_2_result/demo/ so it does not mix with real runs.
"""
from pathlib import Path
import numpy as np
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Result_Report import log_result, load_results, save_results_table, plot_comparison


def main():
    out = Path(__file__).resolve().parents[2] / 'result' / 'stage_2_result' / 'demo'
    log_path = out / 'demo_log.jsonl'
    log_path.unlink(missing_ok=True)

    rng = np.random.default_rng(0)
    true_y = rng.integers(0, 10, size=10000)
    # fake configs with different error rates, standing in for Aaradya's tuning runs
    fake_configs = {
        'baseline_1x128_adam': (0.06, {'hidden': [128], 'optimizer': 'adam', 'lr': 1e-3}),
        '2x256_adam': (0.03, {'hidden': [256, 256], 'optimizer': 'adam', 'lr': 1e-3}),
        '3x512_adam_dropout': (0.02, {'hidden': [512, 512, 512], 'optimizer': 'adam', 'lr': 1e-3, 'dropout': 0.2}),
        '2x256_sgd_lr0.01': (0.05, {'hidden': [256, 256], 'optimizer': 'sgd', 'lr': 1e-2}),
        '2x256_sgd_lr0.1': (0.035, {'hidden': [256, 256], 'optimizer': 'sgd', 'lr': 1e-1}),
    }
    evaluator = Evaluate_Metrics('multiclass metrics', '')
    for name, (error_rate, config) in fake_configs.items():
        pred_y = true_y.copy()
        wrong = rng.random(true_y.size) < error_rate
        pred_y[wrong] = rng.integers(0, 10, size=wrong.sum())
        evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
        log_result(log_path, name, evaluator.evaluate(), config)

    results = load_results(log_path)
    assert list(results) == list(fake_configs)
    save_results_table(results, out / 'demo_table.csv')
    plot_comparison(results, out / 'demo_comparison.png')
    print('Result report checks passed.')


if __name__ == '__main__':
    main()
