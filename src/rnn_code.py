import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary.summary as summary
import pytorch_lightning as pl


def get_params(vocab_size, num_hiddens, device=device):
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
