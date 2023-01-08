# -*- coding: utf-8 -*-
"""Projet d'innovation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CdFzk_KGX_jhOpY4yZ2857lohPpyEP69
"""

from google.colab import files
uploaded = files.upload()

"""# IA

**Code pour entrainer et tester l'IA de reconnaissance d'un mot de passe**
"""

!pip install unidecode

#@title
from __future__ import unicode_literals, print_function, division
from traceback import print_tb
from turtle import done

import unidecode
import unicodedata
import string
import random
import re
import sys

from io import open
import glob
import os
import string
import random

import time
import math

from os import listdir, path, makedirs, popen
from os.path import isdir, isfile, join, basename

import torch
import torch.nn as nn
from torch.autograd import Variable

import time, math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from argparse import ArgumentParser

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA AVAILABLE')
else:
    device = torch.device("cpu")
    print('ONLY CPU AVAILABLE')

n_iters = 100000
all_losses = []
total_loss = 0  # Reset every plot_every iters

n_epochs = 200000
print_every = 10
plot_every = 10
hidden_size = 512
n_layers = 2
lr = 0.005
bidirectional = True

#" .,;'-!@#$%^&*(/\\?*+=)(}{"

#all_letters = string.ascii_letters + string.digits + string.punctuation + string.whitespace
all_letters = string.printable
print('all_letters: ', all_letters)
n_letters = len(all_letters) + 1  # Plus EOS marker


def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip().lower()) for line in some_file]


filename = 'data/passwords/Ashley-Madison.txt'
filenameTrain = 'TrainEval/passwords/PassTrain.txt'
filenameTest = 'TrainEval/passwords/PassTest.txt'


def getLines(f):
    lines = readLines(f)
    print('lines: ', len(lines), ' -> ', f)
    return lines


def split(rate, lines):
    names = []

    # for letter in string.ascii_uppercase:
    for letter in all_letters:
        names_letter = []
        for line in lines:
            if line[0] == letter:
                names_letter.append(line)
            # else:
            #     print("test " + line + " "+letter)
        if len(names_letter) > 0:
            names.append(names_letter)

    print('split names: ', len(names))
    names_traing = []
    names_testing = []
    for names_letter in names:
        length = len(names_letter)

        index = int(length * rate)

        training = names_letter[:index]
        testing = names_letter[index:]

        names_traing.append(training)
        names_testing.append(testing)

    f = open(filenameTrain, "w")
    for names_letter in names_traing:
        for names in names_letter:
            f.write(names + "\n")
    f.close()

    f = open(filenameTest, "w")
    for names_letter in names_testing:
        for names in names_letter:
            f.write(names + "\n")
    f.close()

    return names_traing, names_testing


# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)


def random_training_set(file):
    chunk = random_chunk(file)
    inp = char_tensor(chunk[:-1]).to(device)
    target = char_tensor(chunk[1:]).to(device)
    return inp, target


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTraining(lines):
    line = randomChoice(lines)
    return line


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)  # .long()
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample(lines):
    line = randomTraining(lines)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor


def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = decoder.init_hidden()

    decoder.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = decoder(input_line_tensor[i].to(device), hidden.to(device))
        l = criterion(output.to(device), target_line_tensor[i].to(device))
        loss += l

    loss.backward()

    # decoder_optimizer.step()
    for p in decoder.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return output, loss.item() / input_line_tensor.size(0)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.bidirectional = bidirectional
        self.num_directions = 1
        if self.bidirectional:
            self.num_directions = 2

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1,
                          bidirectional=self.bidirectional, batch_first=True)
        self.out = nn.Linear(self.num_directions * self.hidden_size, output_size)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)

        hidden_concatenated = hidden

        if self.bidirectional:
            hidden_concatenated = torch.cat((hidden[0], hidden[1]), 1)
        else:
            hidden_concatenated = hidden.squeeze(0)

        output = self.out(hidden_concatenated)

        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions, 1, self.hidden_size)

    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))

    def init_hidden_random(self):
        return torch.rand(self.num_directions, 1, self.hidden_size)
    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print('---')
        # print('input: ', input.size())
        # print('hidden: ', hidden.size())
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        # print('output: ', output.size())$
        # input:  torch.Size([1, 59])
        # hidden:  torch.Size([1, 128])
        # output:  torch.Size([1, 59])

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))

    def init_hidden_random(self):
        return torch.rand(1, self.hidden_size)
    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))


def training(n_epochs, lines):
    print()
    print('-----------')
    print('|  TRAIN  |')
    print('-----------')
    print()

    start = time.time()
    all_losses = []
    total_loss = 0
    best_loss = 100
    print_every = n_epochs / 100

    for iter in range(1, n_epochs + 1):
        output, loss = train(*randomTrainingExample(lines))
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f (%.4f)' % (timeSince(start), iter, iter / n_iters * 100, total_loss / iter, loss))


max_length = 20


def samples(start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(start_letter))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSinceStart(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))


def progressPercent(totalNames, start, names, p, samplesGenerated):
    bar_len = 50
    filled_len = int(round(bar_len * names / float(totalNames)))
    percents = round(100.0 * names / float(totalNames), 1)
    nNames = int(p / 100 * totalNames)

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write(
        '[%s] %s%s names founded among %d samples generated (%d of %d names) on %s (goal = %.1f%% = %d names)\r' % (
        bar, percents, '%', samplesGenerated, names, totalNames, timeSinceStart(start), p, nNames))
    sys.stdout.flush()


def progress(total, acc, start, epoch, l):
    bar_len = 50
    filled_len = int(round(bar_len * epoch / float(total)))
    percents = round(100.0 * epoch / float(total), 1)

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s epoch: %d acc: %.3f %% and testing size = %d names => coverage of %.3f %% on %s \r' % (
    bar, percents, '%', epoch, (100 * acc / epoch), l, (100 * acc / l), timeSinceStart(start)))
    sys.stdout.flush()


def sample(decoder, start_letters='ABC'):
    with torch.no_grad():  # no need to track history in sampling

        hidden = decoder.init_hidden_random()

        if len(start_letters) > 1:
            for i in range(len(start_letters)):
                input = inputTensor(start_letters[i])
                # print(start_letters[i], ' ', hidden)
                output, hidden = decoder(input[0].to(device), hidden.to(device))

            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                return start_letters

            letter = all_letters[topi]
            input = inputTensor(letter)
        else:
            input = inputTensor(start_letters)

        output_name = start_letters

        for i in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


def testing(decoder, nb_samples, lineTest, percent):
    print()
    print('------------')
    print('|   TEST   |')
    print('------------')
    print()

    start = time.time()
    accuracy = 0
    predicted = "a"
    predicted_current = []

    if nb_samples > 0:

        for i in range(1, nb_samples + 1):
            # name max size ...
            nc = 1  # random.randint(1, max_length/2 - 1)
            # print('nc: ', nc, ' / ', max_length)

            while predicted in predicted_current:
                starting_letters = ""
                for n in range(nc):
                    rc = random.randint(0, len(string.ascii_uppercase) - 1)
                    starting_letters = starting_letters + string.ascii_uppercase[rc]

                predicted = sample(decoder, starting_letters).lower()

            predicted_current.append(predicted)

            if predicted in lineTest:
                accuracy = accuracy + 1
            # print(starting_letters, ' -> ', predicted, ' accuracy: ', accuracy, ' / ', i , ' = ', (100 * accuracy/i) , ' % and testing corpus contains ', len(lineTest), ' names => coverage of ', (100 * accuracy/len(lineTest)), ' %')

            progress(total=nb_samples, acc=accuracy, start=start, epoch=i, l=len(lineTest))

        accuracy = 100 * accuracy / nb_samples

        print('Accuracy: ', accuracy, '%')

    else:
        i = 0
        l = len(lineTest)
        p = int(percent / 100 * l)
        while accuracy < p:
            nc = random.randint(1, int(max_length / 2 - 1))

            while predicted in predicted_current:
                starting_letters = ""
                for n in range(nc):
                    rc = random.randint(0, len(string.ascii_uppercase) - 1)
                    starting_letters = starting_letters + string.ascii_uppercase[rc]

                predicted = sample(decoder, starting_letters).lower()

            predicted_current.append(predicted)

            if predicted in lineTest:
                accuracy = accuracy + 1
            # print(starting_letters, ' -> ', predicted, ' accuracy: ', accuracy, ' founded over ', l, ' names in ', timeSinceStart(start), ' s and ', i, ' epochs => coverage of ', percent, ' % of total names names = ', p)

            i = i + 1
            progressPercent(totalNames=l, start=start, names=accuracy, p=percent, samplesGenerated=i)

        print(percent + ' % of all names (', len(lineTest), ') reached in ', i, 'iterations (', timeSinceStart(start),
              ' s)...')


def evaluating(decoder):
    print()
    print('------------')
    print('|   EVAL   |')
    print('------------')
    print()

    try:
        while True:
            print('Enter a starting two or tree charachters but less than ', (2 * max_length), ' charachters: ')
            starting_letters = input()
            print()
            if len(starting_letters) > 0 and len(starting_letters) < (2 * max_length):
                print('Generated up to ', max_length, 'charcaters: ')
                predicted = sample(decoder, starting_letters)
                print(predicted)
            else:
                print(starting_letters, ' length < 1 or > ', (2 * max_length))
            print('------------')
            print()

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate evaluating process")
        print('------------')


def getMeanSize(listData):
    mean = 0
    for word in listData:
        mean = mean + len(word)

    return int(mean / len(listData))


if __name__ == '__main__':

    parser = ArgumentParser()
    #
    parser.add_argument("-d", "--trainingData", default="data/passwords/Ashley-Madison.txt", type=str,
                        help="trainingData [path/to/the/data]")
    parser.add_argument("-te", "--trainEval", default='train', type=str, help="trainEval [train, eval, test]")
    #
    parser.add_argument("-r", "--run", default="rnnGeneration", type=str, help="name of the model saved file")
    # parser.add_argument("-mt", "--modelTraining", default='models', type=str, help="Path of the model to save (train) [path/to/the/model]")
    # parser.add_argument("-me", "--modelEval", default='models', type=str, help="Name of the model to load (eval) [path/to/the/model]")
    parser.add_argument("-m", "--model", default='models/rnn.pt', type=str,
                        help="Path of the model to save for trainingof to load for evaluating/testing (eval/test) [path/to/the/model]")
    #
    parser.add_argument('--n', default=10000, type=int,
                        help="number of samples to generate [< 1000]. If < 0, the algorithm will provide names till it reaches the percent (see --p option)")
    parser.add_argument('--ml', default=10, type=int,
                        help="number of characters to generate for each name [default =10]. if < 0 => the number of chars = mean(training set)")
    parser.add_argument('--s', default=0.7, type=int,
                        help="percent of the dataset devoted for training [default =70% and therefore testing =30%]")
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--bidirectional', default=True, type=bool, help="Bidirectionnal model [default True]")
    parser.add_argument('--max_epochs', default=100000, type=int)
    parser.add_argument('-p', '--percent', default=15, type=float,
                        help="percent (number between 1 and 100) of the total names to find (test) [default 15%]")
    #
    args = parser.parse_args()
    #
    repData = args.trainingData  # "data/out/text10.txt"
    # repData = "data/shakespeare.txt"

    file = unidecode.unidecode(open(repData).read())
    file_len = len(file)
    bidirectional = args.bidirectional
    lines = getLines(filename)
    train_set, test_set = split(args.s, lines)

    print('filenameTrain: ', filenameTrain)
    lineTraining = getLines(filenameTrain)
    lineTest = getLines(filenameTest)

    print('lineTraining: ', len(lineTraining))
    print('lineTest: ', len(lineTest))

    if args.ml > 0:
        max_length = args.ml
    else:
        max_length = getMeanSize(lineTraining)

    decoder = RNNLight(n_letters, 128, n_letters).to(
        device)  # RNN(n_characters, args.hidden_size, n_characters, args.num_layers).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    print('decoder: ', decoder)

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_epochs = args.max_epochs

    modelFile = args.run + "_" + str(args.num_layers) + "_" + str(args.hidden_size) + ".pt"

    if not path.exists(args.model):
        makedirs(basename(args.model))

    #########
    # TRAIN #
    #########
    if args.trainEval == 'train':
        decoder.train()
        training(n_epochs, lineTraining)
        torch.save(decoder, args.model)
        print('Model saved in: ', args.model)
    #########
    # EVAL  #
    #########
    elif args.trainEval == 'eval':
        decoder.eval()
        decoder = torch.load(args.model)
        decoder.eval().to(device)
        evaluating(decoder)
    #########
    # TEST  #
    #########
    elif args.trainEval == 'test':
        decoder.eval()
        decoder = torch.load(args.model)
        decoder.eval().to(device)
        testing(decoder, args.n, lineTest, args.percent)
    else:
        print('Choose trainEval option (--trainEval train/eval/test')

from google.colab import drive
drive.mount('/content/drive/')

"""# Occurence de caractères"""

import string

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [(line.strip()) for line in some_file]

all_letters = string.ascii_letters + string.digits + string.punctuation

char_array = list(all_letters)

print(type(char_array))

occur_array = []

"""#Modèle

"""

import tensorflow as tf
# using GRU layers
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    # self.lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)


  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)
    #x = self.lstm(x, training=training)

    if return_state:
      return x, states
    else:
      return x

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.printable
    )

# Using LSTM layers
class MyModelLSTM(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.lstm.get_initial_state(x)
    x, *states = self.lstm(x, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

import numpy 
# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

def getLines(f):
    lines = readLines(f)
    print('lines: ', len(lines), ' -> ', f)
    return lines

def split(rate, lines, filename1, filename2):
    names = []

    # for letter in string.ascii_uppercase:
    for letter in vocab:
        names_letter = []
        for line in lines:
            if line[0] == letter:
                names_letter.append(line)
            # else:
            #     print("test " + line + " "+letter)
        if len(names_letter) > 0:
            names.append(names_letter)

    print('split names: ', len(names))
    names_traing = []
    names_testing = []
    for names_letter in names:
        length = len(names_letter)

        index = int(length * rate)

        training = names_letter[:index]
        testing = names_letter[index:]

        names_traing.append(training)
        names_testing.append(testing)

    f = open(filename1, "w")
    for names_letter in names_traing:
        for names in names_letter:
            f.write(names + "\n")
    f.close()

    f = open(filename2, "w")
    for names_letter in names_testing:
        for names in names_letter:
            f.write(names + "\n")
    f.close()

    return names_traing, names_testing
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def saveDataInFile(data, filename):
  
  np = numpy.array(data)
  print(np)
  with open(filename, 'w') as my_file:
          numpy.savetxt(my_file,np, fmt='%s')
  print('Array exported to file')

"""#Créer un OneStep du modèle"""

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

"""#Créer les corpus

"""

import tensorflow as tf

import numpy as np
import os
import time
import unidecode
import unicodedata
import string
import random
import math
from sklearn.model_selection import train_test_split



lines = getLines('datas.txt')
trainSet, va = train_test_split(lines,test_size=0.3)
#other = getLines('other.txt')
testSet, vaSet = train_test_split(va,test_size=0.5)

saveDataInFile(trainSet, "train.txt")
saveDataInFile(testSet, "test.txt")
saveDataInFile(vaSet, "validation.txt")

!pip install Tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

!pip install -q -U keras-tuner

"""#Entrainement du modèle"""

import tensorflow as tf
import keras_tuner as kt

import numpy as np
import os
import time
import unidecode
import unicodedata
import string
import random
import math
import datetime


# filenameTrain = 'train.txt'
# filenameTest = 'test.txt'

# Read, then decode for py2 compat.
# lines = getLines('datas.txt')
text = open('datas.txt', 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')
vocab = sorted(set(text))
# print(vocab)
print(f'{len(vocab)} unique characters')
# trainSet, va = split(0.7, lines,'train.txt','other.txt')
# other = getLines('other.txt')
# testSet, vaSet = split(0.5, other,'test.txt','validation.txt')
train = open('train.txt', 'rb').read().decode(encoding='utf-8')
test = open('test.txt', 'rb').read().decode(encoding='utf-8')
validation = open('validation.txt', 'rb').read().decode(encoding='utf-8')

# example_texts = ['abcdefg', 'xyz']

chars = sorted(list(set(train)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
# ids = ids_from_chars(chars)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# ids
all_ids = ids_from_chars(tf.strings.unicode_split(train, 'UTF-8'))
all_ids_test = ids_from_chars(tf.strings.unicode_split(test, 'UTF-8'))
all_ids_validation = ids_from_chars(tf.strings.unicode_split(validation, 'UTF-8'))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
ids_dataset_test = tf.data.Dataset.from_tensor_slices(all_ids_test)
ids_dataset_validation = tf.data.Dataset.from_tensor_slices(all_ids_validation)
# for ids in ids_dataset.take(10):
#     print(chars_from_ids(ids).numpy().decode('utf-8'))


seq_length = 100
examples_per_epoch = len(train)//(seq_length+1)
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
sequences_test = ids_dataset_test.batch(seq_length+1, drop_remainder=True)
sequences_validation = ids_dataset_test.batch(seq_length+1, drop_remainder=True)
print("len train: ", len(train))



dataset = sequences.map(split_input_target)
dataset_test = sequences_test.map(split_input_target)
dataset_validation = sequences_validation.map(split_input_target)
# for input_example, target_example in dataset.take(1):
#     print("Input :", text_from_ids(input_example).numpy())
#     print("Target:", text_from_ids(target_example).numpy())

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


dataset_test = (
    dataset_test
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset_validation = (
    dataset_validation
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Length of the vocabulary in chars
vocab_size = len(vocab)

def model_builder(hp):

  # The embedding dimension
  embedding_dim = 256
  hp_dim = hp.Int('embed_units', min_value=64, max_value=512, step=32)

  # Number of RNN units
  rnn_units = 1024
  hp_units = hp.Int('rnn_units', min_value=32, max_value=2048, step=32)


  model = MyModelLSTM(
      # Be sure the vocabulary size matches the `StringLookup` layers.
      vocab_size=len(ids_from_chars.get_vocabulary()),
      embedding_dim=hp_dim,
      rnn_units=hp_units)
  # # model = tf.keras.models.Sequential()
  # # model.add(tf.keras.layers.LSTM(1024, input_shape=(X.shape[1], X.shape[2])))
  # # model.add(tf.keras.layers.Dropout(0.2))
  # # model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=loss, metrics=['accuracy'], run_eagerly=True)

  return model

#for input_example_batch, target_example_batch in dataset.take(1):
    #example_batch_predictions = model(input_example_batch)
    #print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='kerasTuner',
                     project_name='innov')    



# print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
# print()
# print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
#example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("Mean loss:        ", example_batch_mean_loss)
# # tf.exp(example_batch_mean_loss).numpy()
#hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=loss, metrics=['accuracy'], run_eagerly=True)
# Directory where the checkpoints will be saved
checkpoint_dir = 'training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

callback = [
    tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(datetime.datetime.now()),histogram_freq=0, write_graph=True, write_images=True),
    tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True),
]

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

EPOCHS = 20

#[print(i.shape, i.dtype) for i in model.inputs]
#[print(o.shape, o.dtype) for o in model.outputs]
#[print(l.name, l.input_shape, l.dtype) for l in model.layers]
tuner.search(dataset, epochs=EPOCHS, validation_data=dataset_validation, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""The hyperparameter search is complete. embed_units :{best_hps.get('embed_units')}, rnn_units :{best_hps.get('rnn_units')}, optimal learning rate :{best_hps.get('learning_rate')}.""")

model = tuner.hypermodel.build(best_hps)

history = model.fit(dataset, epochs=EPOCHS, validation_data=dataset_validation, callbacks=[callback])

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(dataset, epochs=best_epoch, validation_data=dataset_validation, callbacks=[callback])


# test_scores = model.evaluate(dataset_test, verbose=0)
# print("Test loss:", test_scores[0])
# print("Test accuracy:", test_scores[1])

hypermodel.summary()

print(hypermodel.inputs)

hypermodel.save('saved_model/my_model{}'.format(datetime.datetime.now()))

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/

"""#Test du modèle"""

model.load_weights(checkpoint_dir)

model = hypermodel
 
if tf.test.gpu_device_name():
  print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# tf.saved_model.save(one_step_model, 'one_step')
# one_step_reloaded = tf.saved_model.load('one_step')

# one_step_model = one_step_reloaded
def searchPassword(model, lineTest, percent):

  print()
  print('------------')
  print('|   TEST   |')
  print('------------')
  print()

  max_length = 20
  start = time.time()
  accuracy = 0
  predicted = "a"
  predicted_current = []
  i = 0
  l = len(lineTest)
  p = int(percent / 100 * l)
  asciiChar = string.ascii_letters+string.digits
  generatedPassCount = 1
  pass_finded = []
  while accuracy < p:
      nc = random.randint(2, int(max_length / 2 - 1))

      while predicted in predicted_current:
          starting_letters = ""
          # for n in range(nc):
          #     rc = random.randint(0, len(asciiChar) - 1)
          #     starting_letters = starting_letters + asciiChar[rc]
          
          next_char = tf.constant([str(random.randint(0, len(asciiChar) - 1))]) # , str(random.randint(0, len(asciiChar) - 1))

          states = None
          temp = []
          while  '\n' not in next_char:
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            if ('\n' not in next_char):
              temp.append(next_char)
          if(len(temp) > 1):
            predicted = tf.strings.join(temp)[0].numpy().decode("utf-8")
          # print(predicted)

      predicted_current.append(predicted)
      # print(predicted)
# tf.strings.join(result)[0].numpy().decode("utf-8")
      
      if predicted in lineTest:
          lineTest.remove(predicted)
          accuracy = accuracy + 1
          pass_finded.append(predicted)
          print('->', predicted, ', accuracy:', accuracy, 'founded over', p, 'names in', timeSinceStart(start), 'and', i, 'password generated => coverage of',"{:.4f}".format((accuracy/l)*100), '% of total names =', l,'\n')
      i = i + 1

  saveDataInFile(pass_finded, "generated.txt")
  print(percent, ' % of all names (', len(lineTest), ') reached in ', i, 'iterations (', timeSinceStart(start), ')...')
  
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSinceStart(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))
  

searchPassword(model,test.split(),1)