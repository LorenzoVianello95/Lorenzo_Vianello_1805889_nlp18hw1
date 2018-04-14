import os
import collections
import random
import tensorflow as tf
import numpy as np
import tqdm
import re
import math
from tensorflow.contrib.tensorboard.plugins import projector
from data_preprocessing import build_dataset, save_vectors, read_analogies, generate_batch
from evaluation import evaluation
from gensim.parsing.porter import PorterStemmer

# run on CPU
# comment this part if you want to run it on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### PARAMETERS ###

BATCH_SIZE = 32 #Number of samples per batch.RICoRDA CHE HAI STABILITO CHE BATCH%2*WIND=0
EMBEDDING_SIZE = 280 # Dimension of the embedding vector.
WINDOW_SIZE = 4  # How many words to consider left and right.
NEG_SAMPLES = 180  # Number of negative examples to sample.
VOCABULARY_SIZE = 200#60000 #The most N word to consider in the dictionary

TRAIN_DIR = "dataset/DATA/TRAINM"
VALID_DIR = "dataset/DATA/DEV"
TMP_DIR = "/tmp/"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"


### READ THE TEXT FILES ###

def get_latin(line):
    return ' '.join(''.join([i if ord(i) >=65 and ord(i) <=90 or  ord(i) >= 97 and ord(i) <= 122 else ' ' for i in line]).split())

p = PorterStemmer()

stoplist = set([w.rstrip('\r\n') for w in open("StopwordsM.txt")])
# Read the data into a list of strings.
# the domain_words parameters limits the number of words to be loaded per domain
def read_data(directory, domain_words=-1):
    i=0
    data = []
    for domain in os.listdir(directory):
    #for dirpath, dnames, fnames in os.walk(directory):
        print(str(i)+":"+domain)
        i=i+1
        limit = domain_words
        for f in os.listdir(os.path.join(directory, domain)):
            #print(f)
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f)) as file:
                    for line in file.readlines():
                        #line=line.strip().lower()
                        #line=p.stem_sentence(line)
                        #line=line.lower()
                        line = line.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
                        line=re.sub("(^|\W)\d+($|\W)", " ", line)
                        line = get_latin(line)
                        split = line.lower().strip().split()
                        split = [word for word in split if (word not in stoplist) and (len(word)>1)]
                        if limit > 0 and limit - len(split) < 0:
                            split = split[:limit]
                        else:
                            limit -= len(split)
                        if limit >= 0 or limit == -1:
                            data += split
                        #print(data)
    return data

# load the training set
raw_data = read_data(TRAIN_DIR, domain_words=1000)#40000#ricorda che domainw limita il massimo numero di parole che possono essere lette
print('Data size', len(raw_data))
#print(raw_data)
# the portion of the training set used for data evaluation
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
#valid_examples_m =["german", "most","general", "food", "cat", "eat", "teach"]
#print(valid_examples)


### CREATE THE DATASET AND WORD-INT MAPPING ###

data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)
print('Dictionary size',len(dictionary))
del raw_data  # Hint to reduce memory.
# read the question file for the Analogical Reasoning evaluation
questions = read_analogies(ANALOGIES_FILE, dictionary)
#print(questions)

### MODEL DEFINITION ###

graph = tf.Graph()
eval = None

with graph.as_default():
    # Define input data tensors.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    ### FILL HERE ###
    #definisco le due matrici W1 e W2
    embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0)) #placeholder variable
    i = tf.nn.embedding_lookup(embeddings, train_inputs)
    w = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE],stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
    b = tf.Variable(tf.zeros([VOCABULARY_SIZE]))
    ### FILL HERE ###

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=w,
                           biases=b,
                           labels=train_labels,
                           inputs=i,
                           num_sampled=NEG_SAMPLES,
                           num_classes=VOCABULARY_SIZE))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer =  tf.train.GradientDescentOptimizer(0.3).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

    # evaluation graph
    eval = evaluation(normalized_embeddings, dictionary, questions)

### TRAINING ###

# Step 5: Begin training.
num_steps = 10000001

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR, session.graph)
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    bar = tqdm.tqdm(range(num_steps))
    for step in bar:
        batch_inputs, batch_labels = generate_batch(BATCH_SIZE,2*WINDOW_SIZE, WINDOW_SIZE, data)
        #print({train_inputs: batch_inputs, train_labels: batch_labels})
        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict={train_inputs: batch_inputs, train_labels: batch_labels},
            run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                #print(valid_examples[i],reverse_dictionary[valid_examples[i]])
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                #print(nearest)
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)
        if step % 10000 == 0:
            if step>0:
                eval.eval(session)
                print("avg loss: "+str(average_loss / step))
    final_embeddings = normalized_embeddings.eval()

    ### SAVE VECTORS ###
    #print(final_embeddings)

    save_vectors(final_embeddings)

    # Write corresponding labels for the embeddings.
    with open(TMP_DIR + 'metadata.tsv', 'w') as f:
        for i in range(VOCABULARY_SIZE):
            f.write(reverse_dictionary[i] + '\n')

    with open('dataset/dictionaryBigBig.tsv', 'w') as f:
        for i in range(VOCABULARY_SIZE):
            f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints
    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(TMP_DIR, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()
