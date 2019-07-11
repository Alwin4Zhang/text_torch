import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

'''
RCNN 整体的模型构建流程如下：
　　1）利用Bi-LSTM获得上下文的信息，类似于语言模型。

　　2）将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput, wordEmbedding, bwOutput]。

　　3）将拼接后的向量非线性映射到低维。

　　4）向量中的每一个位置的值都取所有时序上的最大值，得到最终的特征向量，该过程类似于max-pool。

　　5）softmax分类。

'''
class TextRCNN(nn.Module):
    def __init__(self,args,hidden_size_linear=64,hidden_layers=2):
        super(TextRCNN,self).__init__()

        self.args = args

        class_num = args.class_num  # 3, 0 for unk, 1 for negative, 2 for postive
        vocabulary_size = args.vocabulary_size # total number of vocab
        embedding_dimension = args.embedding_dimension # 128
        hidden_size = args.hidden_size

        self.embedding = nn.Embedding(vocabulary_size,embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            embedding_dimension *= 2
        else:
            self.embedding2 = None

        # bi-directional LSTM for Model RCNN
        self.lstm = nn.LSTM(
            input_size = embedding_dimension,
            hidden_size = hidden_size,
            num_layers = hidden_layers,
            dropout = args.dropout,
            bidirectional = True
        )
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(embedding_dimension + 2 * hidden_size, hidden_size_linear)

        # Tanh non-linearity
        self.tanh = nn.Tanh()

        # Fully-Connected Layer
        self.fc = nn.Linear(hidden_size_linear, class_num)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self,x):
        if self.embedding2:
            x = torch.cat([self.embedding(x), self.embedding2(x)], dim=2)  # bz * sen_len * (embed_dim * 2)
            x = x.permute(1, 0, 2)  # Sentence_length(32) * Batch_size * (embed_dim * 2)
        else:
            x = self.embedding(x)  # Batch_size * Sentence_length(32) * embed_dim(128)
            x = x.permute(1, 0, 2)  # Sentence_length(32) * Batch_size embed_dim(128)

        lstm_out, (h_n, c_n) = self.lstm(Variable(x))
        # lstm_out       Sentence_length * Batch_size * (hidden_layers * 2)
        # h_n           （num_layers * 2） * Batch_size * hidden_layers

        input_features = torch.cat([Variable(lstm_out),Variable(x)],dim=2).permute(1, 0, 2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)

        linear_output = self.tanh(self.W(input_features))
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)

        linear_output = linear_output.permute(0, 2, 1)  # Reshaping fot max_pool

        '''
        tf.nn.max_pool(value, ksize, strides, padding, name=None)
        参数是四个，和卷积很类似：

        value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
        ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1] ，stride一般设为2可以将图像缩小一半。
        padding：和卷积类似，可以取'VALID' 或者'SAME'
        '''
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)

        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        return self.softmax(final_out)


