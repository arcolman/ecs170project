"""Run from project root: python -m script.stage_2_script.script_mlp

Trains the final MLP (mlp_configs.FINAL) on the Stage 2 training set and evaluates it on the test set, following the
template pipeline (Dataset_Loader -> Method_MLP -> Result_Saver -> Evaluate_Metrics via Setting_Predefined_Split).
Writes to result/stage_2_result/:
  MLP_prediction_result_None     pickled test predictions (template Result_Saver format)
  mlp_final_history.json         per-epoch training loss / accuracy
  mlp_final_learning_curve.png   training convergence plot for the report (Section 3.5)
and logs the scores to results_log.jsonl / results_table.csv next to the decision tree reference.
"""
from pathlib import Path
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Predefined_Split import Setting_Predefined_Split
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Result_Report import log_result, load_results, save_results_table, save_history, plot_learning_curves
from script.stage_2_script.mlp_configs import FINAL


#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    # Method_MLP seeds torch and numpy itself (seed attribute), so the run is reproducible
    seed = 1
    root = Path(__file__).resolve().parents[2]
    result_folder = root / 'result' / 'stage_2_result'
    log_path = result_folder / 'results_log.jsonl'
    #------------------------------------------------------

    #---- objection initialization setction ---------------
    data_obj = Dataset_Loader('handwritten digits', '10-class digit recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_2_data'

    method_obj = Method_MLP('multi-layer perceptron', '', seed=seed, **FINAL)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_folder) + '/MLP_'
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

    save_history(method_obj.history, result_folder / 'mlp_final_history.json')
    plot_learning_curves({'final MLP': method_obj.history}, result_folder / 'mlp_final_learning_curve.png',
                         title='Final MLP: training loss per epoch')
    log_result(log_path, 'mlp_final', scores, {**FINAL, 'seed': seed})
    save_results_table(load_results(log_path), result_folder / 'results_table.csv')
    print('************ Finish ************')
    #------------------------------------------------------
