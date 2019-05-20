#
# Hyper implementation of word level LSTM language model.
#
# Adapted from https://github.com/pytorch/examples/tree/master/word_language_model
#

import torch
import torch.nn as nn
from torch.nn import functional as F
from layers.hyperlstm import HyperLSTM
from hyper_utils import set_layer_weights


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class LSTMLanguageModel(nn.Module):
    def __init__(self, context_size, block_in, block_out,
                 ntoken=33278, ninp=256, nhid=256, nlayers=2,
                 dropout=0.2, tie_weights=True, batch_size=20,
                 hyper=False):
        super(LSTMLanguageModel, self).__init__()

        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out
        self.hyper = hyper

        self.nlayers = nlayers
        self.nhid = nhid
        self.ntoken = ntoken

        self.hyperlayers = []

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if hyper:
            self.lstm = HyperLSTM(ninp, nhid,
                                  context_size, block_in, block_out,
                                  num_layers=nlayers)
            self.hyperlayers.append(self.lstm)
        else:
            self.lstm = nn.LSTM(ninp, nhid, num_layers=nlayers)

        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.num_projectors = sum([l.num_projectors for l in self.hyperlayers])

        self.init_weights()

        weight = next(self.parameters())
        self.h = (weight.new_zeros(self.nlayers, 1, self.nhid),
                  weight.new_zeros(self.nlayers, 1, self.nhid))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz=1, device=None):
        weight = next(self.parameters())
        if device:
            self.h = (weight.new_zeros(self.nlayers, bsz, self.nhid).to(device),
                      weight.new_zeros(self.nlayers, bsz, self.nhid).to(device))
        else:
            self.h = (weight.new_zeros(self.nlayers, bsz, self.nhid),
                      weight.new_zeros(self.nlayers, bsz, self.nhid))

    def set_weights(self, params):
        assert params.size(0) == self.num_projectors
        set_layer_weights(params, self.hyperlayers)

    def forward(self, input):

        self.h = repackage_hidden(self.h)

        if input.size(1) != self.h[0].size(1):
            print("Init hidden: Correcting batch size")
            if input.is_cuda:
                device = input.get_device()
                print(device)
                self.init_hidden(input.size(1), device)
            else:
                self.init_hidden(input.size(1), None)

        emb = self.drop(self.encoder(input))
        output, self.h = self.lstm(emb, self.h)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)).view(-1, self.ntoken)

if __name__ == '__main__':
    net = LSTMLanguageModel(4, 16, 16, hyper=True)
    print(net)
    print(net.num_projectors)

    net = LSTMLanguageModel(4, 16, 16, hyper=False)
    from model_utils import count_parameters
    print(count_parameters(net))
