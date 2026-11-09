"""CNN configurations for the Stage 3 CIFAR-10 experiments.

BASELINE is the reference model: the MNIST baseline (2 conv blocks of 16 and 32 5x5 filters, max-pool, one fully
connected output layer) adapted to 3x32x32 color images and 10 classes. Each ablation group changes ONE setting from the
baseline (everything else fixed), so any score difference within a group comes from that setting alone.
Any Method_CNN_CIFAR class attribute can be set here.
"""

BASELINE = {
    'conv_channels': (16, 32),
    'kernel_size': 5,
    'pool': 'max',
    'batch_norm': False,
    'fc_dims': (),
    'dropout': 0.0,
    'optimizer_name': 'adam',
    'learning_rate': 1e-3,
    'batch_size': 128,
    'max_epoch': 20,
}

# the CIFAR test set has 10,000 images, so one run is already stable; extra seeds are cheap on a GPU and give the
# mean +/- std format used for ORL. Use (1,) to speed things up if training is slow.
SEEDS = (1, 2, 3)

# group -> {config name: (curve label, settings that differ from BASELINE)}
# at most four configs per group, so each group fits one learning-curve plot
ABLATIONS = {
    'depth': {
        'depth_1': ('1 conv block (16)', {'conv_channels': (16,)}),
        'baseline': ('2 conv blocks (16-32)', {}),
        'depth_3': ('3 conv blocks (16-32-64)', {'conv_channels': (16, 32, 64)}),
        'depth_4': ('4 conv blocks (16-32-64-128)', {'conv_channels': (16, 32, 64, 128)}),
    },
    'batch_norm': {
        'baseline': ('no batch norm', {}),
        'batch_norm': ('batch norm', {'batch_norm': True}),
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


# candidates for the final model (combinations of the changes expected to help, ending with the deeper
# 32-64-128 design with batch norm and dropout); chosen on a validation split held out of the training set by
# script_cnn_cifar_select.py, never by test score. Edit after the ablations if a different combination looks better.
DEEP = {**BASELINE, 'conv_channels': (32, 64, 128), 'kernel_size': 3, 'batch_norm': True, 'fc_dims': (256,), 'dropout': 0.5}
CANDIDATES = {
    'baseline': BASELINE,
    'bn_depth3': {**BASELINE, 'conv_channels': (16, 32, 64), 'batch_norm': True},
    'bn_depth3_k3': {**BASELINE, 'conv_channels': (16, 32, 64), 'kernel_size': 3, 'batch_norm': True},
    'deep_32_64_128': DEEP,
    'deep_32_64_128_k5': {**DEEP, 'kernel_size': 5},
    'deep_32_64_128_4blocks': {**DEEP, 'conv_channels': (32, 64, 128, 256)},
}

# the selected final model, retrained on all 50,000 training images by script_cnn_cifar.py
# PROVISIONAL: update to the winner printed by script_cnn_cifar_select.py (best validation accuracy)
FINAL = CANDIDATES['deep_32_64_128_4blocks']
