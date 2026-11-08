"""CNN configurations for the Stage 3 ORL experiments.

BASELINE is the reference model: the MNIST baseline (2 conv blocks of 16 and 32 5x5 filters, max-pool, one fully
connected output layer) adapted to 1x112x92 faces and 40 classes. Each ablation group changes ONE setting from the
baseline (everything else fixed), so any score difference within a group comes from that setting alone.
Any Method_CNN_ORL class attribute can be set here.
"""

BASELINE = {
    'conv_channels': (16, 32),
    'kernel_size': 5,
    'pool': 'max',
    'fc_dims': (),
    'dropout': 0.0,
    'optimizer_name': 'adam',
    'learning_rate': 1e-3,
    'batch_size': 32,
    'max_epoch': 40,
}

# the ORL test set has only 40 images (1 per person), so one image is 2.5% accuracy:
# every config is run with each of these seeds and reported as mean +/- std
SEEDS = (1, 2, 3)

# group -> {config name: (curve label, settings that differ from BASELINE)}
# at most four configs per group, so each group fits one learning-curve plot
ABLATIONS = {
    'dropout': {
        'baseline': ('no dropout', {}),
        'dropout_0.3': ('dropout 0.3', {'dropout': 0.3}),
        'dropout_0.5': ('dropout 0.5', {'dropout': 0.5}),
    },
    'filters': {
        'filters_8_16': ('8-16 filters', {'conv_channels': (8, 16)}),
        'baseline': ('16-32 filters', {}),
        'filters_32_64': ('32-64 filters', {'conv_channels': (32, 64)}),
    },
    'depth': {
        'depth_1': ('1 conv block (16)', {'conv_channels': (16,)}),
        'baseline': ('2 conv blocks (16-32)', {}),
        'depth_3': ('3 conv blocks (16-32-64)', {'conv_channels': (16, 32, 64)}),
        'depth_4': ('4 conv blocks (16-32-64-128)', {'conv_channels': (16, 32, 64, 128)}),
    },
    'kernel': {
        'kernel_3': ('3x3 kernels', {'kernel_size': 3}),
        'baseline': ('5x5 kernels', {}),
        'kernel_7': ('7x7 kernels', {'kernel_size': 7}),
    },
    'pooling': {
        'baseline': ('max pooling', {}),
        'pool_avg': ('average pooling', {'pool': 'avg'}),
    },
    'optimizer': {
        'baseline': ('Adam (lr 1e-3)', {}),
        'lr_1e-4': ('Adam (lr 1e-4)', {'learning_rate': 1e-4}),
        'opt_sgd_momentum': ('SGD + momentum 0.9 (lr 0.01)', {'optimizer_name': 'sgd_momentum', 'learning_rate': 1e-2}),
    },
}


def ablation_configs():
    '''Every distinct ablation config once: {name: full settings}.'''
    configs = {}
    for group in ABLATIONS.values():
        for name, (_, changes) in group.items():
            configs[name] = {**BASELINE, **changes}
    return configs


# candidates for the final model, filled in after the ablation study (combinations of the changes that helped);
# chosen on a validation split held out of the training set by script_cnn_orl_select.py, never by test score
CANDIDATES = {
    'baseline': BASELINE,
    'dropout_0.5': {**BASELINE, 'dropout': 0.5},
    'k3_dropout_0.5': {**BASELINE, 'kernel_size': 3, 'dropout': 0.5},
    'f32_64_dropout_0.5': {**BASELINE, 'conv_channels': (32, 64), 'dropout': 0.5},
    'f32_64_k3_dropout_0.5': {**BASELINE, 'conv_channels': (32, 64), 'kernel_size': 3, 'dropout': 0.5},
    # the "pool early, small FC layer" design: 3 blocks shrink the map to 64x14x11 before a 128-unit hidden layer
    'depth3_fc128_dropout_0.5': {**BASELINE, 'conv_channels': (16, 32, 64), 'fc_dims': (128,), 'dropout': 0.5},
}

# the selected final model, retrained on all 360 training images by script_cnn_orl.py
# selected by script_cnn_orl_select.py: best mean validation accuracy (99.17% +/- 1.18% over 3 seeds)
FINAL = CANDIDATES['depth3_fc128_dropout_0.5']
