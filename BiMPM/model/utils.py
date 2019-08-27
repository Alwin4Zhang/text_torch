from torchtext import data
from torchtext import datasets
import re
import jieba
import logging

jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


class DataSet(object):
    def __init__(self, args):
        self.RAW = data.Field(sequential=False)
        self.TEXT = data.Field(batch_first=True, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)
        self.TEXT.tokenize = self.word_cut
        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='data',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('id', self.RAW),
                    ('q1', self.TEXT),
                    ('q2', self.TEXT),
                    ('label', self.LABEL)])  # TabularDataset

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=args.vectors)
        self.LABEL.build_vocab(self.train)
        self.RAW.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       # device=args.gpu,
                                       sort_key=sort_key)  # BucketIterator

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        # for <pad>
        self.char_vocab = {'': 0}
        # for <unk> and <pad>
        self.characterized_words = [[0] * self.max_word_len, [0] * self.max_word_len]

        if args.use_char_emb:
            self.build_char_vocab()

    def word_cut(self, text):
        text = regex.sub(' ', text)
        return [word for word in jieba.cut(text) if word.strip()]

    def build_char_vocab(self):
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.characterized_words.append(chars)

    def characterize(self, batch):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.characterized_words[w] for w in words] for words in batch]
