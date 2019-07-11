import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data,datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(test_sen=None):
    # tokenize = lambda x:list(jieba.lcut(x))
    '''
        params:
            fix_length => pad_length
            build_vocab  => building a vocabulary or dictionary mapping all the unique words
            vocab.vectors => shape (vocab_size * embedding_dim) containing the pre-trained word embedding
            BucketIterato => defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    '''
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    print('123')

    # TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    # LABEL.build_vocab(train_data)

    # word_embeddings = TEXT.vocab.vectors
    # print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    # print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    # print ("Label Length: " + str(len(LABEL.vocab)))

    # train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    # train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    # '''Alternatively we can also use the default configurations'''
    # # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    # vocab_size = len(TEXT.vocab)

    # return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter


if __name__ == "__main__":
    load_dataset()