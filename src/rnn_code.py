import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import utils
import helpers


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    ## hidden layer params
    W_xh = torch.randn((num_inputs, num_hiddens), device=device)
    W_hh = torch.randn((num_hiddens, num_hiddens), device=device)
    b_h = torch.zeros(num_hiddens, device=device)

    ## output laayer params
    W_hq = torch.randn((num_hiddens, num_outputs), device=device)
    b_q = torch.zeros(num_outputs, device=device)

    ## Assign gradients to all the params
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)

    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """Returns hidden state at initialization"""
    hidden_state = torch.zeros((batch_size, num_hiddens), device=device)

    ## returning a tuple makes things handle better ar later code
    return (hidden_state,)


def rnn(inputs, state, params):
    """This function defines how to calculate the hidden_state and output_state
    at a given time step.
    NOTE: The RNN model iterates through the outermost dimension of inputs,
    so that it updates the RNN model's hidden state H one-batch at a time,

    Args:
    inputs: shape = [time_steps, batch_size, vocab_size]
    state: the hidden state
    params: model parameters (W_xh, W_hh, b_h, W_hq, b_q)
    """
    (H,) = state
    W_xh, W_hh, b_h, W_hq, b_q = params

    outputs = []

    for xb in inputs:
        ## xb.shape = (batch_size, vocab_size)
        H = torch.tanh(xb @ W_xh + H @ W_hh + b_h)
        y = H @ W_hq + b_q
        outputs.append(y)
    return torch.cat(outputs, dim=0), (H,)


def predict(prefix, num_preds, model, vocab, device="cpu"):
    """Generate new characters following the 'prefix'"""
    ## step-1: get the initial state
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    ## step-2: warmup
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab[y])

    ## step-3: prediction
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])


def grad_clip(model, theta):

    ## step-1: grab the traainable parameters
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params

    ## step-2: calculate the norm of the gradients (i.e. ||g|| )
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))

    ## step-3: clip the gradients
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm


## --------------------------------------------------------------------------------------##


class RNNModelScratch(nn.Module):
    """A RNN model implemented from scratch"""

    def __init__(
        self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn
    ):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.forward_fn = forward_fn
        self.device = device
        self.init_state = init_state
        self.params = get_params(self.vocab_size, self.num_hiddens, self.device)

    def __call__(self, x, state):
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(x, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


class RNNModel(nn.Module):
    """using in-built module of RNN"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.hidden_size = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.directions = 1
        else:
            self.directions = 2
        self.linear = nn.Linear(self.directions * self.hidden_size, self.vocab_size)

    def forward(self, xb, state):
        xb = F.one_hot(xb.T.long(), self.vocab_size).to(torch.float32)
        yb, state = self.rnn(xb, state)

        ## compute the output layer
        yb = yb.reshape((-1, yb.shape[-1]))  ## shape = [time_steps*bs, hidden_size)
        out = self.linear(yb)
        return out, state

    def begin_state(self, batch_size, device):
        """Return hidden state at initilization"""
        if not isinstance(self.rnn, nn.LSTM):
            ## GRU takes a tensor as hidden state init
            state = torch.zeros(
                (
                    self.directions * self.rnn.num_layers,
                    batch_size,
                    self.hidden_size,
                )
            )
        else:
            ## lSTM takes a tuple as hidden state
            state = (
                torch.zeros(
                    (
                        self.directions * self.rnn.num_layers,
                        batch_size,
                        self.hidden_size,
                    ),
                ),
                torch.zeros(
                    (
                        self.directions * self.rnn.num_layers,
                        batch_size,
                        self.hidden_size,
                    )
                ),
            )
        return state


## --------------------------------------------------------------------------------------##


def train_one_epoch(model, train_iter, loss_fn, updater, device, use_random_iter=False):
    """Trains a model for 1 epoch"""
    state = None
    metric = utils.Accumulator(2)

    for xb, yb in train_iter:
        ## ---------------------------------------------------------------------##
        if state is None or use_random_iter:
            ## initialize the state fresh
            ## if it's the 1st iteration or random sampling is used
            state = model.begin_state(batch_size=xb.shape[0], device=device)
        else:
            if isinstance(model, nn.Module) and not isinstance(state, tuple):
                ## state is a TENSOR for nn.GRU
                ## seo, we apply detach_() directly on 'state' variable
                state.detach_()
            else:
                ## state is a TUPLE-of-tensors for nn.LSTM
                ## so, we apply detach_() iteratively on eaach item of 'state'
                for s in state:
                    s.detach_()

        ## ---------------------------------------------------------------------##
        yb = yb.T.reshape(-1).to(device)
        xb = xb.to(device)

        ## forward pass
        y_hat, state = model(xb, state)
        loss = loss_fn(y_hat, yb)

        ## backward pass
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            loss.backward()
            grad_clip(model, 1)
            updater.step()
        else:
            loss.backward()
            grad_clip(model, 1)
            updater(bs=1)
        metric.add(loss * yb.numel(), yb.numel())
    return math.exp(metric[0] / metric[1])


def train(model, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    ...
