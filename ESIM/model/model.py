from torch import nn
import torch
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(self,args):
        super(ESIM,self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.hidden_size = args.hidden_units
        self.embedding_dimension = args.embedding_dimension
        self.vocabulary_size = args.vocab_size
        self.embedding = nn.Embedding(self.vocabulary_size,self.embedding_dimension)
        self.bn_embeds = nn.BatchNorm1d(self.embedding_dimension)
        self.lstm1 = nn.LSTM(
            self.embedding_dimension,
            self.hidden_size,
            batch_first = True,
            bidirectional = True
            )

        self.lstm2 = nn.LSTM(
            self.hidden_size * 8,
            self.hidden_size,
            batch_first = True,
            bidirectional = True
            )
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, 2),
            nn.Softmax(dim=-1)
        )
    
    def soft_attention_align(self,x1,x2,mask1,mask2):
        '''
            x1:batch_size * sequence_length * hidden_size
            x2:batch_size * sequence_length * hidden_size
        '''
        # attention: batch_size * sequence_length * sequence_length
        attention = torch.matmul(x1,x2.transpose(1,2))
        mask1 = mask1.float().masked_fill_(mask1,float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2,float('-inf'))

        # weight: batch_size * sequence_length * sequence_length
        weight1 = F.softmax(attention + mask2.unsqueeze(1),dim=-1)
        x1_align = torch.matmul(weight1,x2)
        weight2 = F.softmax(attention.transpose(1,2) + mask1.unsqueeze(1),dim=-1)
        x2_align = torch.matmul(weight2,x1)
        # x_align: batch_size * sequence_length * hidden_size

        return x1_align,x2_align
    
    def submul(self,x1,x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub,mul],-1)
    
    def apply_multiple(self,x):
        # input: batch_size * sequence_length * ( 2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1,2),x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1,2),x.size(1)).squeeze(-1)
        # output : batch_size * (4 * hidden_size)
        return torch.cat([p1,p2],1)

    def forward(self,*input):
        # batch_size * sequence_length
        sent1,sent2 = input[0],input[1]
        mask1,mask2 = sent1.eq(0),sent2.eq(0)

        # embeds: batch_size * sequence_length => batch_size * sequence_length * embedding_dimension
        x1 = self.bn_embeds(self.embedding(sent1).transpose(1,2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embedding(sent2).transpose(1,2).contiguous()).transpose(1, 2)

        # batch_size * sequence_length * embedding_dimension => batch_size * sequence_length * hidden_size
        o1,_ = self.lstm1(x1)
        o2,_ = self.lstm1(x2)

        # attention
        # batch_size * sequence_length * hidden_size
        q1_align,q2_align = self.soft_attention_align(o1,o2,mask1,mask2)

        # compose
        # batch_size * sequence_length * (8 * hidden_size)
        q1_combined = torch.cat([o1,q1_align,self.submul(o1,q1_align)],-1)
        q2_combined = torch.cat([o2,q2_align,self.submul(o2,q1_align)],-1)

        # batch_size * sequence_length * (2 * hidden_size)
        q1_compose , _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # aggregate
        # input: batch_size * sequence_length * (2 * hidden_size)
        # output: batch_size * ( 4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # similarity
        sim = F.cosine_similarity(q1_rep,q2_rep,dim=1)
        # classifier
        # x = torch.cat([q1_rep,q2_rep],-1)
        # logits = self.fc(x)
        # probabilities = nn.functional.softmax(logits, dim=-1)
        # return logits,probabilities
        return sim