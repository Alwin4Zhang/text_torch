import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numpy.linalg import norm


class TextCNN(nn.Module):
    def __init__(self,args):
        super(TextCNN,self).__init__()
        self.args = args

        channel_num = 1
        filter_num = args.num_kernels
        filter_sizes = args.kernel_sizes

        vocabulary_size = args.vocab_size
        embedding_dimension = args.embedding_dimension
        self.embedding = nn.Embedding(vocabulary_size,embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size,embedding_dimension).from_pretrained(args.vectors)
            channel_num +=1 
        else:
            self.embedding2 = None
        
        self.convs = nn.ModuleList(
            # channel_num,filter_num,(filter_size,embedding_dimension)
            [nn.Conv2d(channel_num,filter_num,(size,embedding_dimension)) for size in filter_sizes]  
        )

        self.dropout = nn.Dropout(args.dropout)
        # self.fc = nn.Linear(len(filter_sizes) * filter_num,class_num)

    def forward(self,x):
        if self.embedding2:
            x = torch.stack([self.embedding(x),self.embedding2(x)],dim=1)
        else:
            x = self.embedding(x) # input.size() = (batch_size, num_seq, embedding_length)  here => (128,sequence_length,embedding_dimension)
            x = x.unsqueeze(1) # input.size() = (batch_size, 1, num_seq, embedding_length)  here => (128,1,sequence_length,embedding_dimension)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # conv_out.size() = (batch_size, filter_num, embedding_dimension, 1) 
        # activation.size() = (batch_size, filter_num, 1)
        x = [F.max_pool1d(item,item.size(2)).squeeze(2) for item in x] # maxpool_out.size() = (batch_size, filter_num)
        x = torch.cat(x,1) # (batch_size, len(filter_sizes)*filter_num)
        x = self.dropout(x) # (batch_size, len(filter_sizes)*filter_num)
        # logits = self.fc(x)
        return x

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
        vocabulary_size = args.vocab_size
        embedding_dimension = args.embedding_dimension
        hidden_size = args.hidden_units

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
        # final_out = self.fc(feature_map)    # Batch_size * class_num  # self.softmax(final_out)
        return feature_map

class SiameseNet(nn.Module):
    def __init__(self,embedding_net):
        super(SiameseNet,self).__init__()
        self.embedding_net = embedding_net
    
    def forward(self,x1,x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        sim = F.cosine_similarity(out1, out2, dim=1)

        return out1,out2,sim

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=0.0):
        super(ContrastiveLoss,self).__init__()
    
    def forward(self,y,y_):
        loss = y * torch.pow(1-y_,2) + (1 - y) * torch.pow(y_-self.margin, 2)
        loss = torch.sum(loss) / 2.0 / len(y)   #y.size()[0]
        return loss
