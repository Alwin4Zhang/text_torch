import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class TextRNN(nn.Module):
    def __init__(self,args,hidden_layers=2):
        super(TextRNN,self).__init__()
        
        self.args = args
        
        '''
        Arguments:
            batch_size: size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
            class_num : 2 = (0,1)
            hidden_size: size of the hidden_state of the LSTM
            vocabulary_size : size of the vocabulary containing unique words
            embedding_dimension: embedding dimension of word
        '''
        class_num = args.class_num
        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dimension
        hidden_size = args.hidden_size

        self.embedding = nn.Embedding(vocabulary_size,embedding_dimension) 
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)

        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            embedding_dimension *= 2
        else:
            self.embedding2 = None

        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            embedding_dimension *= 2
        
        self.lstm = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            # dropout = args.droput,
            bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(hidden_size * hidden_layers * 2, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax()

    
    def forward(self,x):
        if self.embedding2:
            x = torch.cat([self.embedding(x), self.embedding2(x)], dim=2)  # batchsize * sen_len * (embed_dim * 2)
            x = x.permute(1, 0, 2)  # Sentence_length(32) * Batch_size * (embed_dim(128)*2)
        else:
            x = self.embedding(x)       # Batch_size * Sentence_length(32) * embed_dim(128)
            x = x.permute(1, 0, 2)      # Sentence_length(32) * Batch_size * embed_dim(128)
        lstm_out, (h_n, c_n) = self.lstm(Variable(x))
        # lstm_out       Sentence_length * Batch_size * (hidden_layers * 2 [bio-direct])
        # h_n           （num_layers * 2） * Batch_size * hidden_layers
        feature_map = self.dropout(Variable(h_n))  # （num_layers*2） * Batch_size * hidden_layers
        feature_map = torch.cat([feature_map[i, :, :] for i in range(feature_map.shape[0])], dim=1)
        # Batch_size * (hidden_size * hidden_layers * 2)
        final_out = self.fc(feature_map)    # Batch_size * class_num
        return self.softmax(final_out)