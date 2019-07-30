################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.distributions import normal

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        normal_sample = normal.Normal(0, 0.01)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.W_x = nn.Parameter(normal_sample.sample((4*num_hidden, input_dim)))
        self.W_h = nn.Parameter(normal_sample.sample((4*num_hidden, num_hidden)))

        self.W_ph = nn.Parameter(normal_sample.sample((num_classes, num_hidden)))
        self.b_p = nn.Parameter(normal_sample.sample((num_classes,)))

        self.bg = nn.Parameter(normal_sample.sample((num_hidden,)))
        self.bi = nn.Parameter(normal_sample.sample((num_hidden,)))
        self.bf = nn.Parameter(normal_sample.sample((num_hidden,)))
        self.bo = nn.Parameter(normal_sample.sample((num_hidden,)))

        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.nonlins = nn.ModuleList([nn.Tanh(),
                                     nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(),
                                     nn.Tanh()])
        self.to(device)

    def forward(self, x):
        ht_1 = torch.zeros(self.num_hidden, self.batch_size)
        ct_1 = torch.zeros(self.num_hidden, self.batch_size)
        B, seqlen = x.shape
        # for xt in x.t():
        for i in range(self.seq_length-1):
            # efficient implementation from nlp1 project (see paper for explanation)
            # 1 matrix multiplication instead of 4.
            xt = (x[:, i]).unsqueeze(0) # xt = x[:, i, :]
            W_gx, W_ix, W_fx, W_ox = torch.chunk(torch.matmul(self.W_x, xt), chunks=4, dim=0)
            W_gh, W_ih, W_fh, W_oh = torch.chunk(torch.matmul(self.W_h, ht_1), chunks=4, dim=0)

            g = self.nonlins[0](W_gx + W_gh + self.bg[:, None]) # tanh
            i = self.nonlins[1](W_ix + W_ih + self.bi[:, None]) # sigmoid
            f = self.nonlins[2](W_fx + W_fh + self.bf[:, None]) # sigmoid
            o = self.nonlins[3](W_ox + W_oh + self.bo[:, None]) # sigmoid

            c_t = g*i + ct_1 * f          # element-wise
            h_t = self.nonlins[4](c_t)*o  # tanh

            ht_1 = h_t
            ct_1 = c_t

        p_t = torch.matmul(self.W_ph, h_t) + self.b_p[:, None]
        return p_t.t()
