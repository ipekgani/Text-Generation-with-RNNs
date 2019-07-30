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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        normal_sample = normal.Normal(0, 0.0001)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.seq_length = seq_length
        self.Whx = nn.Parameter(normal_sample.sample((num_hidden, input_dim)))
        self.Whh = nn.Parameter(normal_sample.sample((num_hidden, num_hidden)))
        self.Wph = nn.Parameter(normal_sample.sample((num_classes, num_hidden)))
        self.bh = nn.Parameter(normal_sample.sample((num_hidden,)))
        self.bp = nn.Parameter(normal_sample.sample((num_classes,)))

        self.num_hidden = num_hidden
        self.batch_size = batch_size

        self.tanh = nn.Tanh()
        self.to(device)
        # self.softmax = nn.Softmax() cross entropy does it

    def forward(self, x):
        ht_1 = torch.zeros(self.num_hidden, self.batch_size)
        B, seqlen = x.shape
        for i in range(self.seq_length - 1):
            # xt = (x[:, i, :]).t() # only use when onehot
            xt = (x[:, i]).reshape(1, B)
            ht = torch.matmul(self.Whx,xt)
            ht += torch.matmul(self.Whh,ht_1)
            ht += self.bh[:, None]
            ht = self.tanh(ht)
            ht_1 = ht

        pt = torch.matmul(self.Wph, ht) + self.bp[:, None]
        return pt.t()

