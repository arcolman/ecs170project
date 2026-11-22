'''
Concrete Evaluate class for multiclass evaluation metrics
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


def to_numpy(y):
    # Method_MLP returns torch tensors (possibly on a GPU), sklearn methods return numpy arrays
    if hasattr(y, 'detach'):
        y = y.detach().cpu().numpy()
    return np.asarray(y)


class Evaluate_Metrics(evaluate):
    data = None
    # multiclass task, so the binary versions cannot be used
    averages = ('weighted', 'macro', 'micro')

    def evaluate(self):
        print('evaluating performance...')
        true_y = to_numpy(self.data['true_y'])
        pred_y = to_numpy(self.data['pred_y'])
        scores = {'accuracy': accuracy_score(true_y, pred_y)}
        for average in self.averages:
            # zero_division=0 avoids warnings when a class is never predicted (e.g. early in training)
            scores[f'precision_{average}'] = precision_score(true_y, pred_y, average=average, zero_division=0)
            scores[f'recall_{average}'] = recall_score(true_y, pred_y, average=average, zero_division=0)
            scores[f'f1_{average}'] = f1_score(true_y, pred_y, average=average, zero_division=0)
        return scores
