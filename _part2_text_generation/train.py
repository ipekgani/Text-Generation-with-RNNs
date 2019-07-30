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

import os
import time
from datetime import datetime
import argparse
import pickle, glob, os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from _part2_text_generation.dataset import TextDataset
from _part2_text_generation.model import TextGenerationModel

################################################################################

def whatsapp_clean_data(file): # for personal fun
    f = open(file,"r", encoding='utf8')
    newfile = file.split('.')
    newfile = ''.join([newfile[0], '_parsed.txt'])
    newf = open(newfile, "w+", encoding='utf8')
    lines = f.readlines()
    for line in lines:
        newline = line.split('-')
        if len(newline) > 1:
            newline = newline[1].replace('<Medya dahil edilmedi>','')
            newf.write(''.join(newline))
    newf.close()
    f.close()
    return newfile

def onehot(seq_batch, feat_depth):
    newsize = list(seq_batch.shape) # seq x batch
    newsize.append(feat_depth)      # seq x batch x depth
    out = torch.zeros(newsize)
    out.scatter_(2, seq_batch.unsqueeze(-1), 1)
    return out

def get_last_model(folder):
    list_of_files = glob.glob(folder + '/*.mdl')  # * means all if need specific format then *.csv

    if not list_of_files: return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_sample_models(model, dataset, folder):
    Ts = [1e-15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # test temperature
    list_of_files = glob.glob(folder + '/*sample.mdl')  # * means all if need specific format then *.csv
    for file in list_of_files:
        sample_step = int(file.split('_')[-1].split('.')[0].replace('sample',''))
        model.load_state_dict(torch.load(file))
        print('Loading model from step:', sample_step)
        for T in Ts:
            gen_phrase = sequence_generation(model, dataset, temperature=T)
            print('Temperature:', T, ' sequence:', gen_phrase)

def complete_w_last_model(model, config, dataset,
                          query='Sleeping beauty is ', restlen=30, temperature=0.25):

    latest_model = get_last_model(config.model_folder)
    latest_step = 0
    sequence = [dataset._char_to_ix[ch] for ch in query]
    sequence_tensor = torch.tensor(sequence).reshape(len(query),1).detach()

    if latest_model:
        model.load_state_dict(torch.load(latest_model))
        # print(seq_onehot.shape)

        seq_onehot = onehot(sequence_tensor, dataset.vocab_size).permute(1, 0, 2)

        _, (h, c) = model.forward(seq_onehot)
        rest = sequence_generation(model, dataset, next_char=sequence[-1], h=h, c=c,
                                   wantedlen=restlen, temperature=temperature)
        print(query + rest, '<end>')

def continue_from_last_model(model, config, optimizer, scheduler):
    latest_model = get_last_model(config.model_folder)
    latest_step = 0
    if latest_model:
        model.load_state_dict(torch.load(latest_model))
        latest_step = int(latest_model.split('_')[-1].split('.')[0])
        optimizer.load_state_dict(torch.load(config.model_folder + '/optimizer' + str(latest_step) + '.optim'))
        scheduler.load_state_dict(torch.load(config.model_folder + '/scheduler' + str(latest_step) + '.schdlr'))
        print('Loading latest model from iteration {} out of {}'.format(latest_step, config.train_steps))

    return latest_step+1, model, optimizer, scheduler

def sequence_generation(model, dataset, temperature=1, next_char=None, h=None, c=None, wantedlen=None): #default: greedy
    beta = 1/temperature # reciprocal of temperature
    gen_phrase = []
    if wantedlen is None: wantedlen = config.seq_length-1
    if next_char is None:
        next_char = np.random.randint(dataset.vocab_size)
        gen_phrase.append(dataset._ix_to_char[next_char])

    for i in range(wantedlen):
        next_char_onehot = torch.zeros(1, 1, dataset.vocab_size).detach()
        next_char_onehot[:, :, next_char] = 1
        gen_output, (h, c) = model.forward(next_char_onehot, h, c)
        gen_output = gen_output.detach().squeeze()
        gen_output_probs = torch.softmax(beta*gen_output, dim=0)    # exp(beta*E)
        sample_char = torch.multinomial(gen_output_probs, 1).item() # sample
        next_char = sample_char
        # next_char = gen_output.detach().squeeze().argmax().item()  # old version greedy

        gen_phrase.append(dataset._ix_to_char[next_char].replace('\n', '\\n'))

        # gen_phrase.replace('\n', )
    return ''.join(gen_phrase)

def train(config):
    alph_range = (97,123) # vocabulary size! (features)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length,
                                dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, config.device,
                                1-config.dropout_keep_prob).cuda()

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = config.learning_rate_step,
                                                gamma = config.learning_rate_decay)

    latest_step, model, optimizer, scheduler = continue_from_last_model(model, config, optimizer, scheduler)

    log = {'accuracy':[], 'loss':[], 'samples':[]}
    if os.path.isfile(config.model_folder + '/text_gen.pkl'):
        with open(config.model_folder + '/text_gen.pkl', 'rb') as handle:
            log = pickle.load(handle)

    epochs = 0
    while latest_step < config.train_steps: # to support continueing from the last iteration
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            current_batch_size = len(batch_inputs[0])
            if current_batch_size < config.batch_size:
                epochs+=1

            batch_inputs = onehot(torch.stack(batch_inputs, dim=1).to(device), dataset.vocab_size)
            batch_targets = torch.stack(batch_targets, dim=1).to(device)
            # Only for time measurement of step through network
            t1 = time.time()
            output, (_,_) = model.forward(batch_inputs)  # pt
            loss = criterion(output.reshape(current_batch_size*config.seq_length,dataset.vocab_size),
                             batch_targets.flatten())
            loss.backward()


            #######################################################
            # Add more code here ...
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            #######################################################

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            output = output.detach()
            batch_inputs = batch_inputs.detach()
            batch_targets = batch_targets.detach()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = current_batch_size/float(t2-t1+ (1e-7))

            if latest_step % config.print_every == 0:

                predictions = output.argmax(dim=2)
                accuracy = (batch_targets.eq(predictions)).float().mean()

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}, Epoch: {}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), latest_step,
                        config.train_steps, current_batch_size, examples_per_second,
                        accuracy, loss, epochs
                ))

                log['loss'].append(loss.item())
                log['accuracy'].append(accuracy.item())

            if latest_step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                gen_phrase = sequence_generation(model, dataset)
                print('-------------------- Generated sequence:', gen_phrase)
                log['samples'].append(gen_phrase)
                torch.save(model.state_dict(), config.model_folder + '/model_' + str(latest_step) + '.mdl')

            if latest_step % config.save_every == 0 and latest_step != 0:
                torch.save(model.state_dict(), config.model_folder + '/model_' + str(latest_step) + '.mdl')
                torch.save(optimizer.state_dict(), config.model_folder + '/optimizer' + str(latest_step) + '.optim')
                torch.save(scheduler.state_dict(), config.model_folder + '/scheduler' + str(latest_step) + '.schdlr')

                with open(config.model_folder + '/text_gen.pkl', 'wb') as handle:
                    pickle.dump(log, handle)

            if latest_step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
            latest_step += 1

        # convergence
        if len(log['loss'])>6:
            if np.abs(np.mean(log['loss'][-6:-3])-np.mean(log['loss'][-3:])) < 1e-3:
                print('Loss converged.')
                break

    print('Done training.')
    return dataset, model

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='assets/book_agatha.txt',
                        help="Path to a .txt file to train on")  # required=True

    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--save_every', type=int, default=100, help='How often to save model')
    parser.add_argument('--model_folder', type=str, default='saved_models_agatha', help='folder to save/load model')
    parser.add_argument('--temperature', type=float, default=0.75, help='balance greedy & random sampling')  # try 1, 2

    config = parser.parse_args()

    # Train the model
    config.train_steps = int(1e4*4)
    config.sample_every = int(config.train_steps/10)
    config.save_every = int(1e4)

    config.print_every = 50
    config.dropout_keep_prob = 0.8

    # config.txt_file = whatsapp_clean_data(config.txt_file) # for personal fun:P

    os.makedirs(config.model_folder, exist_ok=True)
    config.device = torch.device(torch.cuda.current_device())
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(42)
    np.random.seed(42)
    dataset, model = train(config)

    if os.path.isfile(config.model_folder + '/text_gen.pkl'):
        with open(config.model_folder + '/text_gen.pkl', 'rb') as handle:
            log = pickle.load(handle)
            plotthis = ['loss', 'acc']
            for i,(key, val) in enumerate(list(log.items())):
                if key!='samples':
                    plt.subplot(1, 2, i+1)
                    plt.plot(log[key], label='train', color='b')
                    plt.title(key)
                    if key == 'accuracy':
                        plt.ylim([0, 1.1])
                    plt.legend()
                    plt.xlabel('40000/50 (max steps/print freq)')
                else:
                    # print('Sampled during training:')
                    # for seq in val:
                    #     print(seq)
                    pass
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('0_text_gen.png')

    ''' #######################################################
    Testing different temperature values with the final model #
    ''' #######################################################
    np.random.seed(2323)
    Ts = [1e-15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # test temperature
    for T in Ts:
        gen_phrase = sequence_generation(model, dataset, temperature=T)
        print('Temperature:', T, ' sequence:', gen_phrase)

    '''#######################
    Complete these sentences #
    '''#######################

    print('Completing sentences')
    queries = ['Sleeping beauty is',
               'Murderer was',
               'Poirot used to',
               'Of course it is',
               'His mustasche was',
               'Oh my!',
               'Poirot inspected the',
               'The crime scene is',
               'He said "Detective!\n',
               '\n\n']

    for q in queries:
        complete_w_last_model(model, config, dataset,
                              query=q, restlen=60, temperature=0.2)
    print('Done.\n')

    ''' #############################
    Sample completely new sentences #
    '''##############################

    # print('Getting temperature samples at different iterations')
    # get_sample_models(model, dataset, config.model_folder)