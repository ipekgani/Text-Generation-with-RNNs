# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2,
                 device='cuda:0', dropout=0, batchfirst=True):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        # input size is features
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size

        self.rnn = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden,
                           num_layers=lstm_num_layers, dropout=dropout, batch_first=batchfirst)

        self.output_layer = nn.Linear(self.lstm_num_hidden, self.vocabulary_size,
                                bias = True)

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.lstm_num_layers, x.shape[0], self.lstm_num_hidden)
            c0 = torch.zeros(self.lstm_num_layers, x.shape[0], self.lstm_num_hidden)
        output, (h, c) = self.rnn(x, (h0, c0))
        output = self.output_layer(output)
        return output, (h, c)
