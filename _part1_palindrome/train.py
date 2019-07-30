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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from _part1_palindrome.dataset import PalindromeDataset
from _part1_palindrome.vanilla_rnn import VanillaRNN
from _part1_palindrome.lstm import LSTM
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle, os

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter
plt.style.use('seaborn-darkgrid')

''' SELECT MODEL U WANT TO USE HERE'''
MODEL = VanillaRNN
# MODEL = LSTM

RANGE = (5, 36) # palindrome range
if MODEL==LSTM:
    LR = 0.01
    LR_SCHEDULE = np.array([list(np.ones(2)*LR + 0.005*i) for i in range(20)]).flatten()[:35] #10**(i/4)
else:
    LR = 0.001
    LR_SCHEDULE = list(np.ones((35-5,))*LR)

# print(LR_SCHEDULE)
cmaps = {LSTM.__name__: cm.get_cmap('GnBu'), VanillaRNN.__name__: cm.get_cmap('YlOrRd')}
################################################################################

def onehot(seq_batch, feat_depth):
    newsize = list(seq_batch.shape) # seq x batch
    newsize.append(feat_depth)      # seq x batch x depth
    out = torch.zeros(newsize)
    out.scatter_(2, seq_batch.unsqueeze(-1).long(), 1)
    return out

def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(torch.cuda.current_device())
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(42)

    config.learning_rate = LR_SCHEDULE[config.input_length-5]
    print('Set learning rate:', config.learning_rate)

    # config.input_dim = 1 #config.num_classes # Attention: for one-hot vectors!

    # Initialize the model that we are going to use
    model = MODEL(config.input_length+1, config.input_dim,
                       config.num_hidden, config.num_classes, config.batch_size, device).cuda()

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, end=',')

    log = {'accuracy': [], 'loss': []}
    previous_loss_average = -1
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # batch_inputs = onehot(batch_inputs.to(device), config.num_classes)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        t1 = time.time()
        output = model.forward(batch_inputs)  # pt

        loss = criterion(output, batch_targets)
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why? -> answered in the report!
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()
        optimizer.zero_grad()

        output = output.detach()
        batch_inputs = batch_inputs.detach()
        batch_targets = batch_targets.detach()

        predictions = output.argmax(dim=1)
        accuracy = (batch_targets.eq(predictions)).float().mean()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1 + (1e-7)) #fixme, is this okay??

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            log['loss'].append(loss.item())
            log['accuracy'].append(accuracy.item())

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break


        # for convergence check how much final losses changed (2 sublists at the end from the loss log)
        # the loss is logged every 10 steps so this means we are comparing last two chunks of 30 steps.
        if len(log['loss']) > 100:
            if np.abs(np.mean(log['loss'][-6:-3])-np.mean(log['loss'][-3:])) < 1e-4:
                print('Loss converged.')
                break

    print('Done training.')
    plotthis = ['loss', 'acc']

    for i,(key, val) in enumerate(list(log.items())):
        plt.subplot(1, 2, i+1)
        plt.plot(log[key], label='train', color='b')
        plt.title(key)
        if key == 'accuracy':
            plt.ylim([0, 1.1])
        plt.legend()
    # fig1 = plt.gcf()
    plt.show()
    # plt.draw()
    # fig1.savefig('0_'+ MODEL.__name__ + '.png')

    return log

def experiment(config):
    print('\nPreparing for palindrome experiement')
    p_log = {}
    log_file = 'palindrome_logs.pkl'
    if os.path.isfile(log_file):
        p_log = pickle.load(open(log_file, 'rb'))
        print('Acquiring pickled past training data from', log_file)

    if MODEL.__name__ not in list(p_log.keys()):
        p_log[MODEL.__name__] = {}
        p_log[MODEL.__name__]['log'] = {}

    # get rid of palindrome numbers that we've already ran
    plens = np.arange(*RANGE)
    xticks = (list(np.arange(RANGE[1]-RANGE[0])), plens)
    plens = list(plens)
    numsinlog = [key for key in p_log[MODEL.__name__].keys() if key != 'log']
    for explen in numsinlog:
        if explen in plens:
            plens.pop(plens.index(explen))

    if plens == []:
        print('All palindrome lengths in given range', RANGE, 'for model', MODEL.__name__, 'are already ran. Please change/delete pickle file if you want to re-train.')

    config.train_steps = 4000

    for plen in plens:
        print('Palindrome length', plen)
        config.input_length = plen-1
        log = train(config)
        p_log[MODEL.__name__][plen] = max(log['accuracy'])
        p_log[MODEL.__name__]['log'][plen] = log

        with open('palindrome_submission.pkl', 'wb') as handle:
            pickle.dump(p_log, handle)

    colors = ['r', 'b']
    for i, key in enumerate(list(p_log.keys())):
        acc, x = [], []
        for num in range(35+1):
            if num in p_log[key].keys():
                acc.append(p_log[key][num])
                x.append(num)
        plt.plot(x, acc, label=key, color=colors[i], linewidth=(i+1)*3, marker='o', alpha=0.5)
        plt.legend()

    plt.ylabel('max accuracy')
    plt.xlabel('palindrome length')
    plt.title('max achieved batch acc in training vs. palindrome length')
    plt.ylim([0,1.1])
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('0_' + MODEL.__name__ + '_palindrom.png')

    # for key0 in p_log.keys():
    #     cmap = cmaps[MODEL.__name__]
    #     for num in range(5,40, 3):
    #         if num in p_log[key0]['log'].keys():
    #             log =  p_log[key0]['log'][num]
    #             for i, (key, val) in enumerate(list(log.items())):
    #                 plt.subplot(1, 2, i+1)
    #                 plt.plot(log[key], label='train', color=cmap((40-num)/40), alpha=0.4)
    #                 plt.title(key)
    #                 if key == 'accuracy':
    #                     plt.ylim([0, 1.1])
    # # fig1 = plt.gcf()
    # plt.show()
    # # plt.draw()
    # # fig1.savefig('0_'+ MODEL.__name__ + '.png')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=LR, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    # train(config)
    experiment(config)