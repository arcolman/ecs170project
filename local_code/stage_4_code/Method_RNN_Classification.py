'''
Concrete MethodModule class for a specific learning MethodModule

Recurrent text classifier for the IMDB reviews: word ids in, 2 class logits (negative / positive) out.
Embedding -> recurrent layer(s) -> (Dropout) -> Linear. The recurrent unit is a setting, so the vanilla RNN, LSTM and GRU
models share every other line of code and every other setting, e.g.
  Method_RNN_Classification('rnn', '')                        vanilla RNN (tanh)
  Method_RNN_Classification('lstm', '', rnn_type='lstm')      LSTM, same sizes and training settings
  Method_RNN_Classification('gru', '', rnn_type='gru', bidirectional=True, num_layers=2)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import copy
import time
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


RNN_UNITS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class Method_RNN_Classification(method, nn.Module):
    data = None
    # size of the input/output portals, fixed by the data: vocabulary size in (set from the loaded data), 2 classes out
    vocab_size = None
    output_dim = 2
    # 'rnn' (vanilla, tanh), 'lstm' or 'gru'
    rnn_type = 'rnn'
    embedding_dim = 100
    # optional (vocab_size, embedding_dim) array to start the embedding from, e.g. Dataset_Loader.load_glove(...)
    embedding_weights = None
    freeze_embedding = False
    hidden_dim = 128
    num_layers = 1
    bidirectional = False
    # how the recurrent outputs become one vector per review:
    # 'last' = hidden state after the last real word (padding skipped), 'mean' / 'max' = pooled over all real words
    pooling = 'last'
    # dropout on the embeddings, between stacked recurrent layers, and before the output layer (0 = none)
    dropout = 0.0
    # it defines the max rounds to train the model (one epoch = one pass over all training instances)
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 64
    # 'adam', 'adamw', 'sgd', 'sgd_momentum', 'rmsprop'
    optimizer_name = 'adam'
    weight_decay = 0.0
    # rescale the gradients when their total norm is larger than this (None = off); keeps vanilla RNNs from exploding
    grad_clip = 1.0
    # after training, go back to the weights of the epoch with the best validation accuracy (needs a 'val' split)
    restore_best = True
    # None picks CUDA if available, else CPU (for these small recurrent models the Apple GPU (MPS) is not faster)
    device = None
    seed = 2

    # it defines the the RNN model architecture, e.g.,
    # how many layers, size of variables in each layer, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, **settings):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        for key, value in settings.items():
            if not hasattr(type(self), key):
                raise ValueError(f'unknown RNN setting: {key}')
            setattr(self, key, value)
        if self.rnn_type not in RNN_UNITS:
            raise ValueError(f'unknown rnn_type: {self.rnn_type} (use one of {list(RNN_UNITS)})')
        if self.pooling not in ('last', 'mean', 'max'):
            raise ValueError(f'unknown pooling: {self.pooling}')
        if self.embedding_weights is not None:
            self.vocab_size, self.embedding_dim = np.shape(self.embedding_weights)
        if self.vocab_size is None:
            raise ValueError('vocab_size is required, e.g. vocab_size=len(data["vocab"])')
        # seed before the layers are built, so the initial weights are reproducible
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # word id -> dense vector; id 0 is <pad> and stays a zero vector
        # check here for nn.Embedding doc: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        if self.embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(np.asarray(self.embedding_weights), dtype=torch.float32))
            self.embedding.weight.requires_grad = not self.freeze_embedding
        # check here for nn.RNN doc: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html (LSTM / GRU take the same arguments)
        self.rnn = RNN_UNITS[self.rnn_type](self.embedding_dim, self.hidden_dim, num_layers=self.num_layers,
                                            bidirectional=self.bidirectional, batch_first=True,
                                            dropout=self.dropout if self.num_layers > 1 else 0.0)
        self.drop = nn.Dropout(self.dropout)
        # output layer: no softmax here, nn.CrossEntropyLoss applies log-softmax itself
        self.fc = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.output_dim)

        # filled in by train(), one entry per epoch, used for the learning convergence curves
        self.history = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # it defines the forward propagation function for input x
    def forward(self, x, lengths):
        '''Forward propagation'''
        # x is a n x max_len tensor of word ids, lengths the number of real (non-padding) words of each review
        # the result is a n x output_dim tensor of unnormalized class scores (logits)
        embedded = self.drop(self.embedding(x))
        # packing makes the recurrent layer stop at each review's last real word instead of running over the padding
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.rnn(packed)
        if self.pooling == 'last':
            if self.rnn_type == 'lstm':
                h_n = h_n[0]  # LSTM returns (hidden state, cell state)
            # h_n is (num_layers * directions) x n x hidden_dim; take the top layer (both directions if bidirectional)
            features = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        else:
            out, _ = pad_packed_sequence(packed_out, batch_first=True)  # n x longest x hidden, zeros after each review
            mask = (torch.arange(out.shape[1], device=out.device)[None, :] < lengths[:, None].to(out.device)).unsqueeze(2)
            if self.pooling == 'mean':
                features = (out * mask).sum(1) / lengths.to(out.device)[:, None]
            else:
                features = out.masked_fill(~mask, float('-inf')).max(1).values
        return self.fc(self.drop(features))

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def make_optimizer(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        params = [p for p in self.parameters() if p.requires_grad]
        lr, wd = self.learning_rate, self.weight_decay
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
        raise ValueError(f'unknown optimizer: {self.optimizer_name}')

    def to_tensors(self, split, device):
        return (torch.as_tensor(np.asarray(split['X']), dtype=torch.long, device=device),
                torch.as_tensor(np.asarray(split['lengths']), dtype=torch.long),
                torch.as_tensor(np.asarray(split['y']), dtype=torch.long, device=device))

    # note: nn.Module already has a train(mode) method, which this template method shadows,
    # so we switch between training and evaluation mode with nn.Module.train(self, ...) directly
    def train(self, train, val=None):
        '''train and val are {'X', 'lengths', 'y'} dicts; val (optional) is scored after every epoch.'''
        device = self.get_device()
        self.to(device)
        print(f'training on {device}')
        X, lengths, y = self.to_tensors(train, device)
        n = X.shape[0]

        optimizer = self.make_optimizer()
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # separate generator, so the batch order is reproducible from the seed
        generator = torch.Generator().manual_seed(self.seed)
        best_accuracy, best_state = -1.0, None

        start = time.time()
        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            permutation = torch.randperm(n, generator=generator)
            epoch_loss = torch.zeros((), device=device)
            epoch_correct = torch.zeros((), device=device, dtype=torch.long)
            for i in range(0, n, self.batch_size):
                idx = permutation[i:i + self.batch_size]
                y_pred = self.forward(X[idx.to(device)], lengths[idx])
                # calculate the training loss
                train_loss = loss_function(y_pred, y[idx.to(device)])

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                # accumulate on the device, a .item() per batch would force a slow GPU sync
                epoch_loss += train_loss.detach() * idx.shape[0]
                epoch_correct += (y_pred.detach().argmax(1) == y[idx.to(device)]).sum()

            # mean loss over the epoch's mini-batches (computed while the weights were still changing)
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_loss.item() / n)
            self.history['train_accuracy'].append(epoch_correct.item() / n)
            message = (f'Epoch: {epoch + 1:3d}  Loss: {self.history["train_loss"][-1]:.4f}  '
                       f'Accuracy: {self.history["train_accuracy"][-1]:.4f}')
            if val is not None:
                val_loss, val_accuracy = self.score(val, loss_function, device)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                message += f'  Val loss: {val_loss:.4f}  Val accuracy: {val_accuracy:.4f}'
                if self.restore_best and val_accuracy > best_accuracy:
                    best_accuracy, best_state = val_accuracy, copy.deepcopy(self.state_dict())
                    self.best_epoch = epoch + 1
            print(message + f'  ({time.time() - start:.1f}s)')
        self.method_training_time = time.time() - start

        if best_state is not None:
            self.load_state_dict(best_state)
            print(f'restored the weights of epoch {self.best_epoch} (best validation accuracy {best_accuracy:.4f})')

        # accuracy of the final weights on the whole training set, checked with the template's evaluator
        accuracy_evaluator.data = {'true_y': y.cpu(), 'pred_y': self.predict(X, lengths).cpu()}
        self.final_train_accuracy = accuracy_evaluator.evaluate()
        print('Final training accuracy:', self.final_train_accuracy)

    @torch.no_grad()
    def predict(self, X, lengths, batch_size=1024, return_logits=False):
        # evaluation mode: dropout off
        nn.Module.train(self, False)
        logits = torch.cat([self.forward(X[i:i + batch_size], lengths[i:i + batch_size]) for i in range(0, X.shape[0], batch_size)])
        return logits if return_logits else logits.argmax(1)

    def score(self, split, loss_function, device):
        '''(mean loss, accuracy) on a held-out split with the current weights'''
        X, lengths, y = self.to_tensors(split, device)
        logits = self.predict(X, lengths, return_logits=True)
        return loss_function(logits, y).item(), (logits.argmax(1) == y).float().mean().item()

    def test(self, split):
        # do the testing, and result the result
        device = self.get_device()
        X, lengths, _ = self.to_tensors(split, device)
        start = time.time()
        # instances will get the labels corresponding to the largest logit (same as the largest softmax probability)
        pred_y = self.predict(X, lengths).cpu()
        self.method_testing_time = time.time() - start
        return pred_y

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'], self.data.get('val'))
        print('--start testing...')
        pred_y = self.test(self.data['test'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
