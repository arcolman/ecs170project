"""Run the three MNIST ablations: python -m script.stage_3_script.script_cnn_mnist_ablation.

Each run changes one baseline setting, uses seed 1 and 10 epochs, and saves
separate predictions, history, scores and model weights. Completed runs resume.
The baseline remains the prespecified reference; test scores do not select a model.
"""
import json
from pathlib import Path

import torch

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Result_Report import (
    log_result, load_results, save_results_table, save_history,
    plot_learning_curves, plot_comparison,
)

CONFIGS = {
    'kernel3': {'kernel_size': 3},
    'one_block': {'conv_channels': (16,)},
    'avg_pool': {'pool': 'avg'},
}


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    folder = root / 'result/stage_3_result/mnist'
    folder.mkdir(parents=True, exist_ok=True)
    loader = Dataset_Loader_MNIST('MNIST', '')
    loader.dataset_source_folder_path = root / 'data/stage_3_data'
    data = loader.load()
    for name, changes in CONFIGS.items():
        summary_path = folder / f'{name}_summary.json'
        if summary_path.exists():
            print(f'Skipping completed run: {name}', flush=True)
            continue
        settings = dict(seed=1, device='cpu', **changes)
        model = Method_CNN_MNIST(name, '', **settings)
        print(f'\nStarting {name}: {settings}\n{model}', flush=True)
        model.data = data
        predictions = model.run()
        evaluator = Evaluate_Metrics(name, '')
        evaluator.data = predictions
        scores = evaluator.evaluate()
        save_history(model.history, folder / f'{name}_history.json')
        plot_learning_curves({name: model.history}, folder / f'{name}_learning_curve.png',
                             title=f'MNIST CNN ({name}): training loss per epoch')
        torch.save({'state_dict': model.state_dict(), 'settings': settings}, folder / f'{name}_model.pt')
        torch.save(predictions, folder / f'{name}_predictions.pt')
        log_result(folder / 'results_log.jsonl', name, scores, settings)
        save_results_table(load_results(folder / 'results_log.jsonl'), folder / 'results_table.csv')
        table_path = folder / 'results_table.csv'
        table_path.write_bytes(table_path.read_bytes().replace(b'\r\n', b'\n'))
        summary = {
            'name': name, 'settings': settings, 'epochs': model.max_epoch,
            'learning_rate': model.learning_rate, 'batch_size': model.batch_size,
            'optimizer': model.optimizer_name, 'loss': model.loss_name,
            'training_seconds': model.method_training_time,
            'final_train_accuracy': model.final_train_accuracy,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'scores': scores, 'torch_version': torch.__version__,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        del model
    names = ['baseline', *CONFIGS]
    histories = {name: json.loads((folder / f'{name}_history.json').read_text()) for name in names}
    scores = load_results(folder / 'results_log.jsonl')
    plot_learning_curves(histories, folder / 'ablation_learning_curves.png',
                         title='MNIST: one-setting-at-a-time ablations')
    plot_comparison({name: scores[name] for name in names}, folder / 'ablation_comparison.png')
    print('All three MNIST ablations completed.', flush=True)


if __name__ == '__main__':
    main()
