"""MLP configurations for the Stage 2 experiments.

BASELINE is the reference model. Each ablation group changes ONE setting from the baseline (everything else fixed),
so any score difference within a group comes from that setting alone. Any Method_MLP class attribute can be set here.
"""

BASELINE = {
    'hidden_dims': (256, 128),
    'activation': 'relu',
    'dropout': 0.0,
    'optimizer_name': 'adam',
    'learning_rate': 1e-3,
    'batch_size': 128,
    'max_epoch': 20,
    'loss_name': 'cross_entropy',
}

# group -> {config name: (curve label, settings that differ from BASELINE)}
# at most four configs per group, so each group fits one learning-curve plot
ABLATIONS = {
    'depth': {
        'depth_1': ('1 hidden layer (256)', {'hidden_dims': (256,)}),
        'baseline': ('2 hidden layers (256-128)', {}),
        'depth_3': ('3 hidden layers (512-256-128)', {'hidden_dims': (512, 256, 128)}),
        'depth_4': ('4 hidden layers (512-256-128-64)', {'hidden_dims': (512, 256, 128, 64)}),
    },
    'width': {
        'width_64_32': ('64-32', {'hidden_dims': (64, 32)}),
        'baseline': ('256-128', {}),
        'width_1024_512': ('1024-512', {'hidden_dims': (1024, 512)}),
    },
    'activation': {
        'baseline': ('ReLU', {}),
        'act_tanh': ('Tanh', {'activation': 'tanh'}),
        'act_sigmoid': ('Sigmoid', {'activation': 'sigmoid'}),
    },
    'optimizer': {
        'baseline': ('Adam (lr 1e-3)', {}),
        'opt_sgd': ('SGD (lr 0.01)', {'optimizer_name': 'sgd', 'learning_rate': 1e-2}),
        'opt_sgd_momentum': ('SGD + momentum 0.9 (lr 0.01)', {'optimizer_name': 'sgd_momentum', 'learning_rate': 1e-2}),
        'opt_rmsprop': ('RMSprop (lr 1e-3)', {'optimizer_name': 'rmsprop'}),
    },
    'learning_rate': {
        'lr_1e-4': ('Adam lr 1e-4', {'learning_rate': 1e-4}),
        'baseline': ('Adam lr 1e-3', {}),
        'lr_1e-2': ('Adam lr 1e-2', {'learning_rate': 1e-2}),
    },
    'loss': {
        'baseline': ('Cross-entropy', {}),
        'loss_label_smoothing': ('Cross-entropy, label smoothing 0.1', {'label_smoothing': 0.1}),
        'loss_mse': ('MSE on softmax vs one-hot', {'loss_name': 'mse'}),
    },
    'batch_size': {
        'batch_32': ('batch 32', {'batch_size': 32}),
        'baseline': ('batch 128', {}),
        'batch_512': ('batch 512', {'batch_size': 512}),
        'batch_2048': ('batch 2048', {'batch_size': 2048}),
    },
    'regularization': {
        'baseline': ('none', {}),
        'dropout_0.2': ('dropout 0.2', {'dropout': 0.2}),
        'dropout_0.5': ('dropout 0.5', {'dropout': 0.5}),
        'weight_decay_1e-4': ('weight decay 1e-4', {'optimizer_name': 'adamw', 'weight_decay': 1e-4}),
    },
    'initialization': {
        'baseline': ('PyTorch default (Kaiming uniform, a=sqrt(5))', {}),
        'init_kaiming': ('Kaiming normal (He)', {'init_name': 'kaiming'}),
        'init_xavier': ('Xavier uniform (Glorot)', {'init_name': 'xavier'}),
    },
}

# repeat every configuration with these seeds and report mean +/- std, since many differences are only ~0.1-0.3%
SEEDS = (1, 2, 3)


# candidates for the final model: combinations of the single changes that helped most in the ablation
# (label smoothing, dropout 0.2, 3 hidden layers). script_mlp_select.py picks one on a validation split carved from
# the training set, so the test set is not used to choose the final model.
CANDIDATES = {
    'ls': {**BASELINE, 'label_smoothing': 0.1},
    'ls_dropout': {**BASELINE, 'label_smoothing': 0.1, 'dropout': 0.2},
    'deep_ls_dropout': {**BASELINE, 'hidden_dims': (512, 256, 128), 'label_smoothing': 0.1, 'dropout': 0.2},
    'deep_ls_dropout_cosine30': {**BASELINE, 'hidden_dims': (512, 256, 128), 'label_smoothing': 0.1, 'dropout': 0.2,
                                 'lr_schedule': 'cosine', 'max_epoch': 30},
    'wide_ls_dropout_cosine30': {**BASELINE, 'hidden_dims': (1024, 512), 'label_smoothing': 0.1, 'dropout': 0.3,
                                 'lr_schedule': 'cosine', 'max_epoch': 30},
}

# the final model reported in Sections 3.2 / 3.5 / 3.6, trained by script_mlp.py on all 60,000 training instances:
# best validation accuracy in script_mlp_select.py (98.60% +/- 0.03%, see result/stage_2_result/selection/)
FINAL = CANDIDATES['deep_ls_dropout_cosine30']
