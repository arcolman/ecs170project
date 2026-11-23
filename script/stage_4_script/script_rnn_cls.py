"""Run from project root: python -m script.stage_4_script.script_rnn_cls [run_name]

Trains a recurrent classifier on the IMDB training reviews (minus the fixed validation slice) and evaluates it once on
the test set, following the template pipeline
(Dataset_Loader_IMDB -> Method_RNN_Classification -> Result_Saver -> Evaluate_Metrics via Setting_Predefined_Split).

Runs are defined in RUNS below; the default is the vanilla RNN baseline. Person C: add 'lstm' / 'gru' runs that only change
rnn_type, so every other setting stays the same as the RNN baseline, and put architecture changes in their own runs.
Writes to result/stage_4_result/classification/:
  <run>_prediction_result_None   pickled test predictions (template Result_Saver format)
  <run>_history.json             per-epoch training / validation loss and accuracy
  cls_<run>_curves.png           learning curves (report Section 3.5)
  cls_<run>_confusion.png        test confusion matrix
and logs the scores to results_log.jsonl / results_table.csv, one row per run.
"""
from pathlib import Path
import sys
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader_IMDB
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Predefined_Split import Setting_Predefined_Split
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_4_code.Result_Report import (log_result, load_results, save_results_table, save_history,
                                                   plot_train_val_curves, plot_confusion_matrix)

# shared baseline settings (Method_RNN_Classification defaults, written out so the report can quote them):
# 100-d embedding trained from scratch, 1 layer of 128 hidden units, last hidden state -> linear, Adam lr 1e-3,
# batch 64, 10 epochs, gradient clipping at 1.0, weights of the best validation epoch kept
RUNS = {
    'rnn': {'rnn_type': 'rnn'},
    'lstm': {'rnn_type': 'lstm'},
    'gru': {'rnn_type': 'gru'},
    # one-change-at-a-time variants of the vanilla RNN
    'rnn_mean_pool': {'rnn_type': 'rnn', 'pooling': 'mean'},
}
TITLES = {'rnn': 'Vanilla RNN', 'lstm': 'LSTM', 'gru': 'GRU'}


#---- Recurrent Neural Network script ----
if 1:
    #---- parameter section -------------------------------
    run_name = sys.argv[1] if len(sys.argv) > 1 else 'rnn'
    settings = RUNS[run_name]
    seed = 1
    root = Path(__file__).resolve().parents[2]
    result_folder = root / 'result' / 'stage_4_result' / 'classification'
    result_folder.mkdir(parents=True, exist_ok=True)
    log_path = result_folder / 'results_log.jsonl'
    #------------------------------------------------------

    #---- objection initialization setction ---------------
    data_obj = Dataset_Loader_IMDB('IMDB', 'binary sentiment classification of movie reviews')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_4_data' / 'text_classification'

    # the vocabulary size is only known after loading, so load here and hand the data to the setting
    # (the setting's load() call then returns the same data without reading the 50,000 files again)
    loaded = data_obj.load()
    data_obj.load = lambda: loaded
    method_obj = Method_RNN_Classification(run_name, '', vocab_size=len(loaded['vocab']), seed=seed, **settings)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_folder) + f'/{run_name}_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Predefined_Split('predefined train/test split', '')

    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    #------------------------------------------------------

    #---- running section ---------------------------------
    print('************ Start ************')
    print(method_obj)
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for metric, value in scores.items():
        print(f'{metric:>20}: {value:.4f}')
    print(f'training time: {method_obj.method_training_time:.1f}s, final training accuracy: {method_obj.final_train_accuracy:.4f}')

    title = TITLES.get(run_name, run_name)
    save_history(method_obj.history, result_folder / f'{run_name}_history.json')
    plot_train_val_curves(method_obj.history, result_folder / f'cls_{run_name}_curves.png',
                          title=f'IMDB {title}: loss and accuracy per epoch')
    plot_confusion_matrix(result_obj.data['true_y'], result_obj.data['pred_y'], result_folder / f'cls_{run_name}_confusion.png',
                          class_names=('negative', 'positive'), title=f'{title}, test set')
    config = {**settings, 'seed': seed, 'best_epoch': getattr(method_obj, 'best_epoch', None),
              'training_time_s': round(method_obj.method_training_time, 1)}
    log_result(log_path, run_name, scores, config)
    save_results_table(load_results(log_path), result_folder / 'results_table.csv')
    print('************ Finish ************')
    #------------------------------------------------------
