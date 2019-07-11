import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,args):
        super(TextCNN,self).__init__()
        self.args = args
        
        class_num = args.class_num
        channel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
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
        self.fc = nn.Linear(len(filter_sizes) * filter_num,class_num)

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
        logits = self.fc(x)
        return logits