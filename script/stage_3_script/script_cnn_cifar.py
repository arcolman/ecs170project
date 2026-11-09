"""Run from project root: python -m script.stage_3_script.script_cnn_cifar [baseline|final]

Trains the CIFAR CNN (cifar_configs.FINAL by default, or cifar_configs.BASELINE) on the 50,000 training images and evaluates it
on the 10,000 test images, following the template pipeline
(Dataset_Loader_CIFAR -> Method_CNN_CIFAR -> Result_Saver -> Evaluate_Metrics via Setting_Predefined_Split).
Writes to result/stage_3_result/cifar/:
  CNN_<run>_prediction_result_None   pickled test predictions (template Result_Saver format)
  <run>_history.json                 per-epoch training loss, training accuracy and test accuracy
  <run>_learning_curve.png           training convergence plot for the report (Section 3.5)
  <run>_train_vs_test_accuracy.png   training vs test accuracy per epoch (overfitting check)
  <run>_seeds.json                   the same config with every seed in cifar_configs.SEEDS (mean +/- std)
and logs the main-seed scores to results_log.jsonl / results_table.csv, one row per run (same format as MNIST).
"""
from pathlib import Path
import contextlib
import io
import json
import sys
import numpy as np
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_CIFAR
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Predefined_Split import Setting_Predefined_Split
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Result_Report import log_result, load_results, save_results_table, save_history, plot_learning_curves
from script.stage_3_script.cifar_configs import BASELINE, FINAL, SEEDS


#---- Convolutional Neural Network script ----
if 1:
    #---- parameter section -------------------------------
    run_name = sys.argv[1] if len(sys.argv) > 1 else 'final'
    settings = {'baseline': BASELINE, 'final': FINAL}[run_name]
    assert settings is not None, 'set FINAL in cifar_configs.py first (see script_cnn_cifar_select.py)'
    seed = SEEDS[0]
    root = Path(__file__).resolve().parents[2]
    result_folder = root / 'result' / 'stage_3_result' / 'cifar'
    result_folder.mkdir(parents=True, exist_ok=True)
    log_path = result_folder / 'results_log.jsonl'
    #------------------------------------------------------

    #---- objection initialization setction ---------------
    data_obj = Dataset_Loader_CIFAR('CIFAR', '10-class object recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_3_data'

    method_obj = Method_CNN_CIFAR('convolutional neural network', '', seed=seed, **settings)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_folder) + f'/CNN_{run_name}_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Predefined_Split('predefined train/test split', '')

    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    #------------------------------------------------------

    #---- running section ---------------------------------
    print('************ Start ************')
    print(method_obj)
    print(f'{sum(p.numel() for p in method_obj.parameters()):,} trainable parameters')
    # test accuracy per epoch is only recorded for the overfitting plot; it is not used for training or selection
    method_obj.monitor_data = data_obj.load()['test']
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for metric, value in scores.items():
        print(f'{metric:>20}: {value:.4f}')
    print(f'training time: {method_obj.method_training_time:.1f}s, final training accuracy: {method_obj.final_train_accuracy:.4f}')

    save_history(method_obj.history, result_folder / f'{run_name}_history.json')
    plot_learning_curves({run_name: method_obj.history}, result_folder / f'{run_name}_learning_curve.png',
                         title=f'CIFAR CNN ({run_name}): training loss per epoch')
    h = method_obj.history
    plot_learning_curves({'training accuracy (per-epoch, during training)': {'epoch': h['epoch'], 'acc': h['train_accuracy']},
                          'test accuracy (after each epoch)': {'epoch': h['epoch'], 'acc': h['monitor_accuracy']}},
                         result_folder / f'{run_name}_train_vs_test_accuracy.png', key='acc', ylabel='Accuracy',
                         title=f'CIFAR CNN ({run_name}): training vs test accuracy')
    log_result(log_path, run_name, scores, {**{k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()},
                                            'seed': seed, 'train_accuracy': float(method_obj.final_train_accuracy)})
    save_results_table(load_results(log_path), result_folder / 'results_table.csv')

    # repeat with the other seeds to report mean +/- std (same format as ORL)
    per_seed = {seed: {**{k: float(v) for k, v in scores.items()}, 'train_accuracy': float(method_obj.final_train_accuracy)}}
    data = data_obj.load()
    for other_seed in SEEDS[1:]:
        m = Method_CNN_CIFAR('convolutional neural network', '', seed=other_seed, **settings)
        m.data = data
        e = Evaluate_Metrics('multiclass metrics', '')
        with contextlib.redirect_stdout(io.StringIO()):
            e.data = m.run()
            s = e.evaluate()
        per_seed[other_seed] = {**{k: float(v) for k, v in s.items()}, 'train_accuracy': float(m.final_train_accuracy)}
        print(f'seed {other_seed}: test accuracy {s["accuracy"]:.4f}, weighted F1 {s["f1_weighted"]:.4f}')
    summary = {k: {'mean': float(np.mean([r[k] for r in per_seed.values()])),
                   'std': float(np.std([r[k] for r in per_seed.values()]))} for k in per_seed[seed]}
    (result_folder / f'{run_name}_seeds.json').write_text(json.dumps({'per_seed': per_seed, 'summary': summary}, indent=2))
    print(f"over seeds {SEEDS}: test accuracy {summary['accuracy']['mean']:.4f} +/- {summary['accuracy']['std']:.4f}, "
          f"weighted F1 {summary['f1_weighted']['mean']:.4f} +/- {summary['f1_weighted']['std']:.4f}")
    print('************ Finish ************')
    #------------------------------------------------------
