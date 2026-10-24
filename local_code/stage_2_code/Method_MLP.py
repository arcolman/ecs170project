'''
Concrete MethodModule class for a specific learning MethodModule

MLP for the Stage 2 data: 784 pixel features in, 10 class logits out.
Every architecture / training choice is a class attribute, so a tuning run only changes settings, not code.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import time
import torch
from torch import nn
import numpy as np


ACTIVATIONS = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}


class Method_MLP(method, nn.Module):
    data = None
    # hidden layer sizes, one entry per hidden layer, e.g. (256, 128) is 784 -> 256 -> 128 -> 10
    hidden_dims = (256, 128)
    # 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu'
    activation = 'relu'
    # dropout probability after each hidden activation (0 = no dropout)
    dropout = 0.0
    batch_norm = False
    # size of the input/output portals, fixed by the data: 28x28 pixels in, digits 0-9 out
    input_dim = 784
    output_dim = 10
    # it defines the max rounds to train the model (one epoch = one pass over all training instances)
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # 60,000 instances is too many for one full batch, so we train on shuffled mini-batches
    batch_size = 128
    # 'adam', 'sgd', 'sgd_momentum', 'rmsprop', 'adagrad'
    optimizer_name = 'adam'
    weight_decay = 0.0
    # 'cross_entropy' or 'mse' (MSE between the softmax output and the one-hot label)
    loss_name = 'cross_entropy'
    label_smoothing = 0.0
    # 'none' or 'cosine' (cosine decay of the learning rate down to 0 over max_epoch)
    lr_schedule = 'none'
    # 'default' (PyTorch's nn.Linear init), 'kaiming' or 'xavier'
    init_name = 'default'
    # None picks CUDA if available, else CPU; set 'cpu', 'cuda' or 'mps' to force one
    device = None
    seed = 2

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    # any class attribute above can be overridden by keyword, e.g. Method_MLP('mlp', '', hidden_dims=(512,), optimizer_name='sgd')
    def __init__(self, mName, mDescription, **settings):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        for key, value in settings.items():
            if not hasattr(Method_MLP, key):
                raise ValueError(f'unknown MLP setting: {key}')
            setattr(self, key, value)
        # seed before the layers are built, so the initial weights are reproducible
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # each hidden layer: Linear -> (BatchNorm) -> activation -> (Dropout)
        layers = []
        in_dim = self.input_dim
        for out_dim in self.hidden_dims:
            # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            layers.append(nn.Linear(in_dim, out_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(ACTIVATIONS[self.activation]())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        # output layer: no softmax here, nn.CrossEntropyLoss applies log-softmax itself
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)
        self.init_weights()

        # filled in by train(), one entry per epoch, used for the learning convergence curve
        self.history = {'epoch': [], 'train_loss': [], 'train_accuracy': []}

    def init_weights(self):
        if self.init_name == 'default':
            return
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if self.init_name == 'kaiming':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif self.init_name == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    raise ValueError(f'unknown init: {self.init_name}')
                nn.init.zeros_(layer.bias)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # x is a n x 784 tensor, the result is a n x 10 tensor of unnormalized class scores (logits)
        return self.layers(x)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        # MPS (Apple GPU) works with device='mps', but for an MLP this small it was ~4x slower than the CPU
        # on our M-series Mac (per-batch launch overhead outweighs the tiny matrix multiplies), so it is not the default
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
        # move the whole dataset to the device once, then slice mini-batches out of it
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
        n = X.shape[0]

        optimizer = self.make_optimizer()
        loss_function = self.make_loss_function()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch) if self.lr_schedule == 'cosine' else None
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
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                # accumulate on the device, a .item() per batch would force a slow GPU sync
                epoch_loss += train_loss.detach() * idx.shape[0]
                epoch_correct += (y_pred.detach().argmax(1) == y[idx]).sum()
            if scheduler is not None:
                scheduler.step()

            # mean loss over the epoch's mini-batches (computed while the weights were still changing)
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_loss.item() / n)
            self.history['train_accuracy'].append(epoch_correct.item() / n)
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == self.max_epoch:
                print(f'Epoch: {epoch + 1:3d}  Loss: {self.history["train_loss"][-1]:.4f}  '
                      f'Accuracy: {self.history["train_accuracy"][-1]:.4f}  ({time.time() - start:.1f}s)')
        self.method_training_time = time.time() - start

        # accuracy of the final weights on the whole training set, checked with the template's evaluator
        accuracy_evaluator.data = {'true_y': y.cpu(), 'pred_y': self.predict(X).cpu()}
        self.final_train_accuracy = accuracy_evaluator.evaluate()
        print('Final training accuracy:', self.final_train_accuracy)

    @torch.no_grad()
    def predict(self, X, batch_size=4096):
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
