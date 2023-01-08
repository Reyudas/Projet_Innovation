import string
import numpy
import os
import time
import unidecode
import unicodedata
import string
import random
import math

import tensorflow as tf

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


def saveDataInFile(data, filename):
  
  np = numpy.array(data)
  print(np)
  with open(filename, 'w') as my_file:
          numpy.savetxt(my_file,np, fmt='%s')
  print('Array exported to file') 

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


test = open('test.txt', 'rb').read().decode(encoding='utf-8')

print("1 : GRU model")
print("2 : LSTM model")

input_value = int(input("Wich model do you want to run: "))

print(input_value)
while (input_value != 1 and input_value !=2):
  print("Wrong value")
  input_value = int(input("Wich model do you want to run: "))

if(input_value == 1):
  model = tf.keras.models.load_model('modelGRU')
else :
  model = tf.keras.models.load_model('modelLSTM')  
    

text = open('./data/datas.txt', 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))

ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)

chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

searchPassword(model,test.split(),1)