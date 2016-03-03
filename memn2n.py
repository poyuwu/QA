'''Train a memory network on the bAbI dataset.

References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''

from __future__ import print_function
from keras.models import Sequential,Graph
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge, Permute, Dropout,RepeatVector,Reshape,Flatten,TimeDistributedMerge
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
from keras.layers import Layer
import tarfile
import numpy as np
import re
from keras import backend as K

class TimeDistributedMerge2D(Layer):
    input_ndim = 3
    def __init__(self, mode='sum', dims='1',**kwargs):

        super(TimeDistributedMerge2D, self).__init__(**kwargs)
        self.mode = mode
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        self.dims = int(dims)
    @property
    def output_shape(self):
        return_list = list(self.input_shape)
        return_list.pop(self.dims)
        return tuple(return_list)#(None,self.input_shape[1] ,self.input_shape[3])

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.mode == 'ave':
            s = K.mean(X, axis=self.dims)
            return s
        if self.mode == 'sum':
            s = K.sum(X, axis=self.dims)
            return s
        elif self.mode == 'mul':
            s = K.prod(X, axis=self.dims)
            return s
        else:
            raise Exception('Unknown merge mode')
class TimeDistributedMerge3D(Layer):
    input_ndim = 4
    def __init__(self, mode='sum', dims=1,**kwargs):

        super(TimeDistributedMerge3D, self).__init__(**kwargs)
        self.mode = mode
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        self.dims = int(dims)
    @property
    def output_shape(self):
        return_list = list(self.input_shape)
        return_list.pop(self.dims)
        return tuple(return_list)#(None,self.input_shape[1] ,self.input_shape[3])

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.mode == 'ave':
            s = K.mean(X, axis=self.dims)
            return s
        if self.mode == 'sum':
            s = K.sum(X, axis=self.dims)
            return s
        elif self.mode == 'mul':
            s = K.prod(X, axis=self.dims)
            return s
        else:
            raise Exception('Unknown merge mode')

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(story, q, answer) for story,q,answer in data]#[(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen,story_maxnum):
    X = []
    Xq = []
    Y = []
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    for story, query, answer in data:
        x = []
        for i in range(story_maxnum-len(story)):
            x.append([])
        for facts in story:
            x.append([word_idx[w] for w in facts])
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(pad_sequences(x,maxlen=story_maxlen))
        Xq.append(xq)
        Y.append(y)
    return (np.array(X),
            pad_sequences(Xq, maxlen=story_maxlen), np.array(Y))


path = get_file('babi-tasks-v1-2.tar.gz',
                origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type] # challenge = 'single_supporting_fact_10k'

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))
flatten = lambda data: reduce(lambda x, y: x + y, data)
vocab = sorted(reduce(lambda x, y: x | y, (set(flatten(story) + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxnum = max(map(len, (x for x, _, _ in train_stories + test_stories)))
story_maxlen = max(map(len, (y for y in [x for x, _, _ in train_stories + test_stories])))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
"""
print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Story max number:', story_maxnum, 'facts')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')
"""
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen,story_maxnum)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen,story_maxnum)
inputs_test.resize((1000,100))
inputs_train.resize((10000,100))
"""
print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
"""
print('Compiling...')
# embed the input sequence into a sequence of vectors

input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64,
                              input_length=story_maxlen*story_maxnum),
                         )
input_encoder_m.add(Reshape(dims=(10,10,64)))
input_encoder_m.add(TimeDistributedMerge3D(dims=2))
input_encoder_m.add(Dropout(0.1))
# output: (samples, story_maxlen*story_maxnum, embedding_dim)
# output: (None, 100, 64)
# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=story_maxlen))
# output: (None,story_maxlen* embedding_dim )
question_encoder.add(Dropout(0.1))
question.add(TimeDistributedMerge())
# compute a 'match' between input sequence elements (which are vectors)
# and the question vector sequence
match = Sequential()
match.add(Merge([input_encoder_m, question_encoder],
                mode='dot',
                dot_axes=[(2,), (2,)]))
#match.add(Permute((2, 1)))
#match.add(Reshape(dims=(10,10,64)))
#match.add(TimeDistributedMerge2D(dims=2))
match.add(Activation('softmax'))
# output: (samples, story_maxlen, query_maxlen)
# embed the input into a single vector with size = story_maxlen:
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=64,
                              input_length=story_maxlen*story_maxnum))
input_encoder_c.add(Dropout(0.1))
input_encoder_c.add(Reshape(dims=(10,10,64)))
input_encoder_c.add(TimeDistributedMerge3D(dims=2))
# output: (samples, story_maxlen, embedding_dim)
# output: (samples, story_maxlen, query_maxlen)
# sum the match vector with the input vector:
response = Sequential()
response.add(Merge([match, input_encoder_c],
                    mode='dot',
                    dot_axes=[(1,), (1,)]))
#response.add(TimeDistributedMerge3D(dims=1))
# output: (samples, story_maxlen, query_maxlen)
#response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)
#response.add(Reshape(dims=(1,64)))
# concatenate the match vector with the question vector,
# and do logistic regression on top
#question_encoder.add(TimeDistributedMerge3D(mode='ave',dims=1))

answer = Sequential()
answer.add(Merge([response, question_encoder], mode='sum'))#, concat_axis=-1))
# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer.add(LSTM(32))
# one regularization layer -- more would probably be needed.
answer.add(Dropout(0.3))
answer.add(Dense(vocab_size))
# we output a probability distribution over the vocabulary
answer.add(Activation('softmax'))
answer.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note: you could use a Graph model to avoid repeat the input twice
answer.fit([inputs_train, queries_train, inputs_train], answers_train,
           batch_size=32,
           nb_epoch=600,
           show_accuracy=True,
           validation_data=([inputs_test, queries_test, inputs_test], answers_test))
