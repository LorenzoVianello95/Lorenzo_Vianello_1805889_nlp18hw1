import collections
import random
import numpy as np


### generate_batch ###
# This function generates the train data and label batch from the dataset.
#
### Parameters ###
# batch_size: the number of train_data,label pairs to produce per batch
# curr_batch: the current batch number.
# window_size: the size of the context
# data: the dataset
### Return values ###
# train_data: train data for current batch
# labels: labels for current batch
#TODO: not put in batch couple like (x,x)
def generate_batch(batch_size, data_index, window_size, data ):
  dim_4word = 2 * window_size
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  total_window_size = 2 * window_size + 1  # [ window_size target window_size ]
  step = batch_size // dim_4word
  c = collections.deque(maxlen=total_window_size)
  data_index = (data_index*step) % len(data)
  for _ in range(total_window_size):
    #print(data_index)
    c.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(step):
    t = window_size  # target label at the center of the buffer
    avoid = [window_size]
    for j in range(dim_4word):
      while t in avoid:
        t = random.randint(0, total_window_size - 1)
      avoid.append(t)
      batch[i * dim_4word + j] = c[window_size]
      labels[i * dim_4word + j, 0] = c[t]
    c.append(data[data_index])
  return batch, labels




### build_dataset ###
# This function is responsible of generating the dataset and dictionaries.
# While constructing the dictionary take into account the unseen words by
# retaining the rare (less frequent) words of the dataset from the dictionary
# and assigning to them a special token in the dictionary: UNK. This
# will train the model to handle the unseen words.
### Parameters ###
# words: a list of words
# vocab_size:  the size of vocabulary
#
### Return values ###
# data: list of codes (integers from 0 to vocabulary_size-1).
#       This is the original text but words are replaced by their codes
# dictionary: map of words(strings) to their codes(integers)
# reverse_dictionary: maps codes(integers) to words(strings)
def build_dataset(words, vocab_size):
    dictionary = dict()
    reversed_dictionary = dict()
    data = []
    freqDict={}

    for wrd in words:
        if freqDict.has_key(wrd):
            c=freqDict.get(wrd)
            freqDict[wrd] = c + 1
        else:
            freqDict[wrd]=1

    listW=sorted(freqDict, key=freqDict.get, reverse=True)
    #print(freqDict,listW)
    dictionary["UNK"]=0
    for index in range(1,vocab_size):
        if listW:
            mom=listW.pop(0)
            dictionary[mom]=index
    #print(dictionary)
    reversed_dictionary = {v: k for k, v in dictionary.iteritems()}
    #print(reversed_dictionary)

    for wrd in words:
        if dictionary.has_key(wrd):
            c=dictionary.get(wrd)
            data.append(c)
        else:
            data.append(0)
    #print(data)
    ###FILL HERE###

    return data, dictionary, reversed_dictionary

###
# Save embedding vectors in a suitable representation for the domain identification task
###
def save_vectors(vectors):
    with open('dataset/embeddingWBigBig.tsv', 'w') as f:
        i=0
        for vect in list(vectors):
            f.write(str(vect) + '_')
            i=i+1
        #print(i)

    ###FILL HERE###



# Reads through the analogy question file.
#    Returns:
#      questions: a [n, 4] numpy array containing the analogy question's
#                 word ids.
#      questions_skipped: questions skipped due to unknown words.
#
def read_analogies(file, dictionary):
    questions = []
    questions_skipped = 0
    with open(file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip())) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)
