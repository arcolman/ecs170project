"""Run from project root: python -m script.stage_2_script.test_mlp

Checks Method_MLP on small synthetic data, so it does not need the dataset (runs in a few seconds).
"""
import numpy as np
import torch
from local_code.stage_2_code.Method_MLP import Method_MLP


def synthetic_data(n, seed):
    # 10 well-separated clusters in 784 dimensions with features in [0,1], like the scaled pixels
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(0).random((10, 784))
    y = rng.integers(0, 10, size=n)
    X = np.clip(centers[y] + rng.normal(0, 0.1, size=(n, 784)), 0, 1).astype(np.float32)
    return X, y.astype(np.int64)


def main():
    X_train, y_train = synthetic_data(2000, seed=1)
    X_test, y_test = synthetic_data(500, seed=2)
    data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

    # output portal: 10 unnormalized logits per instance
    model = Method_MLP('mlp', '', hidden_dims=(32, 16))
    assert model.forward(torch.zeros(5, 784)).shape == (5, 10)

    # every loss / optimizer / regularization option trains, the loss goes down, and the clusters are learned
    for settings in ({}, {'loss_name': 'mse', 'learning_rate': 1e-2}, {'optimizer_name': 'sgd_momentum', 'learning_rate': 1e-2},
                     {'dropout': 0.3, 'batch_norm': True, 'init_name': 'kaiming', 'lr_schedule': 'cosine'}):
        model = Method_MLP('mlp', '', hidden_dims=(32, 16), max_epoch=5, batch_size=64, **settings)
        model.data = data
        result = model.run()
        accuracy = (result['pred_y'].numpy() == y_test).mean()
        print(f'{settings}: loss {model.history["train_loss"][0]:.3f} -> {model.history["train_loss"][-1]:.3f}, test accuracy {accuracy:.3f}')
        assert len(model.history['train_loss']) == 5
        assert model.history['train_loss'][-1] < model.history['train_loss'][0]
        assert accuracy > 0.9

    # test() runs in evaluation mode, so dropout is off and predictions are deterministic
    assert torch.equal(model.test(X_test), model.test(X_test))
    assert not model.training

    # same seed -> same initial weights and same training run
    a, b = (Method_MLP('mlp', '', hidden_dims=(32,), max_epoch=2, seed=7) for _ in range(2))
    for m in (a, b):
        m.train(X_train, y_train)
    assert a.history == b.history

    # misspelled settings fail loudly instead of being silently ignored
    try:
        Method_MLP('mlp', '', hiden_dims=(32,))
        raise AssertionError('unknown setting was accepted')
    except ValueError:
        pass

    print('MLP checks passed.')


if __name__ == '__main__':
    main()
