"""Run from project root: python -m script.stage_3_script.script_cnn_mnist

Trains the baseline CNN (2 conv blocks + 1 fully connected output layer) on the MNIST training set and evaluates it on
the test set, following the template pipeline
(Dataset_Loader_MNIST -> Method_CNN_MNIST -> Result_Saver -> Evaluate_Metrics via Setting_Predefined_Split).
Writes to result/stage_3_result/mnist/:
  CNN_prediction_result_None   pickled test predictions (template Result_Saver format)
  <run>_history.json           per-epoch training loss / accuracy
  <run>_learning_curve.png     training convergence plot for the report (Section 3.5)
and logs the scores to results_log.jsonl / results_table.csv, one row per run.
Config experiments: change run_name and settings below, e.g. settings = {'kernel_size': 3}.
"""
from pathlib import Path
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Predefined_Split import Setting_Predefined_Split
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Result_Report import log_result, load_results, save_results_table, save_history, plot_learning_curves


#---- Convolutional Neural Network script ----
if 1:
    #---- parameter section -------------------------------
    run_name = 'baseline'
    # anything not listed uses the Method_CNN_MNIST class defaults (2 conv blocks of 16 and 32 5x5 filters, max-pool, 1 FC)
    settings = {}
    seed = 1
    root = Path(__file__).resolve().parents[2]
    result_folder = root / 'result' / 'stage_3_result' / 'mnist'
    result_folder.mkdir(parents=True, exist_ok=True)
    log_path = result_folder / 'results_log.jsonl'
    #------------------------------------------------------

    #---- objection initialization setction ---------------
    data_obj = Dataset_Loader_MNIST('MNIST', '10-class handwritten digit recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_3_data'

    method_obj = Method_CNN_MNIST('convolutional neural network', '', seed=seed, **settings)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_folder) + '/CNN_'
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

    save_history(method_obj.history, result_folder / f'{run_name}_history.json')
    plot_learning_curves({run_name: method_obj.history}, result_folder / f'{run_name}_learning_curve.png',
                         title=f'MNIST CNN ({run_name}): training loss per epoch')
    log_result(log_path, run_name, scores, {**settings, 'seed': seed})
    save_results_table(load_results(log_path), result_folder / 'results_table.csv')
    print('************ Finish ************')
    #------------------------------------------------------
