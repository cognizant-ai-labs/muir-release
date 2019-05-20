#
# Class for hyperconv 2d layers in pytorch
#

import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from hyper_utils import pack_conv2d

class HyperConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 context_size, block_in, block_out,
                 frozen_context=False,
                 bias=True,
                 stride=1,
                 padding=1):
        super(HyperConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out
        self.stride = stride
        self.padding = 1

        # Create context vectors
        assert (in_channels % block_in == 0) and (out_channels % block_out == 0)
        self.num_block_rows = out_channels / block_out
        self.num_block_cols = in_channels / block_in
        self.num_blocks = int(self.num_block_rows * self.num_block_cols *
                                 self.kernel_size * self.kernel_size)
        self.num_projectors = self.num_blocks

        if frozen_context:
            self.context = torch.zeros(self.num_blocks, context_size, 1)
        else:
            self.context = Parameter(torch.Tensor(self.num_blocks, context_size, 1))

        # Create bias vector.
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        initial_context = math.sqrt(2) / math.sqrt(
                                self.in_channels * self.kernel_size * self.kernel_size)
        init.constant_(self.context, initial_context)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_weight(self, params):
        self.weight = pack_conv2d(params,
                                  self.in_channels,
                                  self.out_channels,
                                  self.kernel_size)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias,
                        padding=self.padding,
                        stride=self.stride)

if __name__ == "__main__":
    print(HyperConv2d(10, 20, 3, 4, 5, 4))

