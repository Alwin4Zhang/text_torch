import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class AttentionModel(nn.Module):
    def __init__(self,args,hidden_layers=2):
        super(AttentionModel,self).__init__()

        batch_size = args.batch_size
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
            dropout = args.dropout,
            bidirectional=True)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2,hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size,1)
        )

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(hidden_size * 2, class_num)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2))

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def attention_net(self,lstm_output):
        ''' 
        self attention
        '''
        M = self.tanh(lstm_output) # [batch_size,sequence_length,hidden_size * 2]
        a = F.softmax(torch.matmul(M,self.w),dim=1).unsqueeze(-1) # [batch_size,sequence_length, 1]
        out = lstm_output * a  # [batch_size,sequence_length, hidden_size * 2]
        out = torch.sum(out,1) #[batch_size, hidden_size * 2]
        return out
        
    def self_attention(self,lstm_output):
        energy = self.projection(lstm_output)
        weights = F.softmax(energy.squeeze(-1),dim=1)
        outputs = (lstm_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs

    def forward(self,x):
        if self.embedding2:
            x = torch.cat([self.embedding(x), self.embedding2(x)], dim=2)  # batchsize * sen_len * (embed_dim * 2)
            x = x.permute(1, 0, 2)  # Sentence_length(32) * Batch_size * (embed_dim(128)*2)
        else:
            x = self.embedding(x)       # Batch_size * Sentence_length(32) * embed_dim(128)
            x = x.permute(1, 0, 2)      # Sentence_length(32) * Batch_size * embed_dim(128)
        lstm_out, (h_n, c_n) = self.lstm(Variable(x))
        # lstm_out       Sentence_length * Batch_size * (hidden_size * 2 [bio-direct])
        # h_n           （num_layers * 2） * Batch_size * hidden_layers
        lstm_out = lstm_out.permute(1,0,2) # batch_size * sentence_length * hidden_size
        # attn_out = self.attention_net(Variable(lstm_out))  # [batch_size,sequence_length, hidden_size * 2]
        attn_out = self.self_attention(Variable(lstm_out))
        output = self.fc(attn_out)
        output = self.softmax(output)
        return output

