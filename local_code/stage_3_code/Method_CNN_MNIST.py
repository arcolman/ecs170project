'''
Concrete MethodModule class for a specific learning MethodModule

Baseline CNN for MNIST: 1x28x28 images in, 10 class logits out.
conv blocks (Conv2d -> ReLU -> pooling) followed by fully connected layers. Every architecture / training choice is
a class attribute, so a config experiment only changes settings, not code, e.g.
Method_CNN_MNIST('cnn', '', conv_channels=(32, 64, 64), kernel_size=3, pool='avg').
Method_CNN_ORL / Method_CNN_CIFAR can start from a copy of this file: change in_channels, input_size, output_dim.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import time
import torch
from torch import nn
import numpy as np


ACTIVATIONS = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}
POOLS = {'max': nn.MaxPool2d, 'avg': nn.AvgPool2d}


class Method_CNN_MNIST(method, nn.Module):
    data = None
    # size of the input/output portals, fixed by the data: 1 x 28 x 28 grayscale digits in, digits 0-9 out
    in_channels = 1
    input_size = (28, 28)
    output_dim = 10
    # number of filters of each conv block, one entry per block, e.g. (16, 32) is 1 -> 16 -> 32 channels
    conv_channels = (16, 32)
    kernel_size = 5
    # None keeps the spatial size through each conv ("same" padding for odd kernels)
    padding = None
    stride = 1
    # 'max' or 'avg', applied after each conv block with this window (2 halves height and width)
    pool = 'max'
    pool_size = 2
    batch_norm = False
    # hidden fully connected layer sizes after flattening; () means the flattened features go straight to the output
    fc_dims = ()
    # 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu'
    activation = 'relu'
    # dropout probability before each fully connected layer (0 = no dropout)
    dropout = 0.0
    # it defines the max rounds to train the model (one epoch = one pass over all training instances)
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 128
    # 'adam', 'adamw', 'sgd', 'sgd_momentum', 'rmsprop', 'adagrad'
    optimizer_name = 'adam'
    weight_decay = 0.0
    # 'cross_entropy' or 'mse' (MSE between the softmax output and the one-hot label)
    loss_name = 'cross_entropy'
    label_smoothing = 0.0
    # None picks CUDA, then MPS (Apple GPU), else CPU; set 'cpu', 'cuda' or 'mps' to force one
    device = None
    seed = 2

    # it defines the the CNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, **settings):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        for key, value in settings.items():
            if not hasattr(type(self), key):
                raise ValueError(f'unknown CNN setting: {key}')
            setattr(self, key, value)
        # seed before the layers are built, so the initial weights are reproducible
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # each conv block: Conv2d -> (BatchNorm) -> activation -> pooling
        padding = self.kernel_size // 2 if self.padding is None else self.padding
        blocks = []
        in_channels = self.in_channels
        for out_channels in self.conv_channels:
            # check here for nn.Conv2d doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            blocks.append(nn.Conv2d(in_channels, out_channels, self.kernel_size, stride=self.stride, padding=padding))
            if self.batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(ACTIVATIONS[self.activation]())
            blocks.append(POOLS[self.pool](self.pool_size))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        # flattened feature size depends on kernel/padding/stride/pooling, so measure it with a dummy image
        # (eval mode, since batch norm cannot train on a single image)
        self.features.eval()
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, self.in_channels, *self.input_size)).numel()
        self.features.train()

        # fully connected layers: (Dropout) -> Linear -> activation, then the output layer
        layers = [nn.Flatten()]
        in_dim = flat_dim
        for out_dim in self.fc_dims:
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers += [nn.Linear(in_dim, out_dim), ACTIVATIONS[self.activation]()]
            in_dim = out_dim
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        # output layer: no softmax here, nn.CrossEntropyLoss applies log-softmax itself
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.classifier = nn.Sequential(*layers)

        # filled in by train(), one entry per epoch, used for the learning convergence curve
        self.history = {'epoch': [], 'train_loss': [], 'train_accuracy': []}

    # it defines the forward propagation function for input x
    def forward(self, x):
        '''Forward propagation'''
        # x is a n x C x H x W tensor, the result is a n x output_dim tensor of unnormalized class scores (logits)
        return self.classifier(self.features(x))

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        # unlike the Stage 2 MLP, convolutions are heavy enough that the Apple GPU beats the CPU
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def make_optimizer(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        params, lr, wd = self.parameters(), self.learning_rate, self.weight_decay
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'sgd_momentum':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
        if self.optimizer_name == 'adagrad':
            return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
        raise ValueError(f'unknown optimizer: {self.optimizer_name}')

    def make_loss_function(self):
        if self.loss_name == 'cross_entropy':
            # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        if self.loss_name == 'mse':
            mse = nn.MSELoss()
            return lambda logits, y: mse(torch.softmax(logits, dim=1), nn.functional.one_hot(y, self.output_dim).float())
        raise ValueError(f'unknown loss: {self.loss_name}')

    # note: nn.Module already has a train(mode) method, which this template method shadows,
    # so we switch between training and evaluation mode with nn.Module.train(self, ...) directly
    def train(self, X, y):
        device = self.get_device()
        self.to(device)
        print(f'training on {device}')
        # move the whole dataset to the device once, then slice mini-batches out of it
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
        n = X.shape[0]

        optimizer = self.make_optimizer()
        loss_function = self.make_loss_function()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # separate generator, so the batch order is reproducible from the seed
        generator = torch.Generator().manual_seed(self.seed)

        start = time.time()
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            permutation = torch.randperm(n, generator=generator).to(device)
            epoch_loss = torch.zeros((), device=device)
            epoch_correct = torch.zeros((), device=device, dtype=torch.long)
            for i in range(0, n, self.batch_size):
                idx = permutation[i:i + self.batch_size]
                y_pred = self.forward(X[idx])
                # calculate the training loss
                train_loss = loss_function(y_pred, y[idx])

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                # accumulate on the device, a .item() per batch would force a slow GPU sync
                epoch_loss += train_loss.detach() * idx.shape[0]
                epoch_correct += (y_pred.detach().argmax(1) == y[idx]).sum()

            # mean loss over the epoch's mini-batches (computed while the weights were still changing)
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_loss.item() / n)
            self.history['train_accuracy'].append(epoch_correct.item() / n)
            print(f'Epoch: {epoch + 1:3d}  Loss: {self.history["train_loss"][-1]:.4f}  '
                  f'Accuracy: {self.history["train_accuracy"][-1]:.4f}  ({time.time() - start:.1f}s)')
        self.method_training_time = time.time() - start

        # accuracy of the final weights on the whole training set, checked with the template's evaluator
        accuracy_evaluator.data = {'true_y': y.cpu(), 'pred_y': self.predict(X).cpu()}
        self.final_train_accuracy = accuracy_evaluator.evaluate()
        print('Final training accuracy:', self.final_train_accuracy)

    @torch.no_grad()
    def predict(self, X, batch_size=1024):
        # evaluation mode: dropout off, batch norm uses its running statistics
        nn.Module.train(self, False)
        return torch.cat([self.forward(X[i:i + batch_size]).argmax(1) for i in range(0, X.shape[0], batch_size)])

    def test(self, X):
        # do the testing, and result the result
        device = self.get_device()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        start = time.time()
        # instances will get the labels corresponding to the largest logit (same as the largest softmax probability)
        pred_y = self.predict(X).cpu()
        self.method_testing_time = time.time() - start
        return pred_y

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
