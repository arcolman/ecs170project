"""Run from project root: python -m script.stage_2_script.script_decision_tree

Decision tree on the Stage 2 train/test split, scored with Evaluate_Metrics. This checks the full
pipeline on real data before the MLP is ready, and gives a non-neural reference point for the report.
"""
from pathlib import Path
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_DT import Method_DT
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Predefined_Split import Setting_Predefined_Split
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Result_Report import log_result, load_results, save_results_table
import numpy as np


#---- Decision Tree script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(1)
    root = Path(__file__).resolve().parents[2]
    result_folder = root / 'result' / 'stage_2_result'
    log_path = result_folder / 'results_log.jsonl'
    #------------------------------------------------------

    #---- objection initialization setction ---------------
    data_obj = Dataset_Loader('handwritten digits', '10-class digit recognition')
    data_obj.dataset_source_folder_path = root / 'data' / 'stage_2_data'

    method_obj = Method_DT('decision tree', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_folder) + '/DT_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Predefined_Split('predefined train/test split', '')

    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')
    #------------------------------------------------------

    #---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores, _ = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for metric, value in scores.items():
        print(f'{metric:>20}: {value:.4f}')

    log_result(log_path, 'decision_tree', scores, {'method': 'sklearn DecisionTreeClassifier', 'params': 'default'})
    save_results_table(load_results(log_path), result_folder / 'results_table.csv')
    print('************ Finish ************')
    #------------------------------------------------------
