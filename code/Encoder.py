import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
import os.path as osp

class Conv1dSamePad(nn.Module):
    def __init__(self, in_channels, out_channels, filter_len, stride=1, **kwargs):
        super(Conv1dSamePad, self).__init__()
        self.filter_len = filter_len
        self.conv = nn.Conv1d(in_channels, out_channels, filter_len, padding=(self.filter_len // 2), stride=stride,
                              **kwargs)
        nn.init.xavier_normal_(self.conv.weight)
        # nn.init.constant_(self.conv.bias, 1 / out_channels)

    def forward(self, x):
        if self.filter_len % 2 == 1:
            return self.conv(x)
        else:
            return self.conv(x)[:, :, :-1]

class FastText(nn.Module):
    '''
    FastText实现
    '''
    def __init__(self, vocab_size, num_classes):
        super(FastText, self).__init__()
        emb_dim = 256
        hidden_size = 300
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_size)
        self.last_dim = hidden_size
        self.fc = nn.Linear(self.last_dim, num_classes)

    def forward(self, data):
        '''前向传播到输出层之前'''
        x = data.x # 只取text部分的特征
        x = self.embedding(x)
        x = F.dropout(x, training=self.training)
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        return x
    
    def out(self, data):
        '''前向传播到输出层'''
        x = self.forward(data)
        x = F.dropout(x, training=self.training)
        out = self.fc(x)
        return out

class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(TextCNN, self).__init__()
        emb_dim = 256
        filter_num = 100
        filter_sizes = [3, 4, 5]
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, kernel_size=(w, emb_dim)) for w in filter_sizes])
        self.last_dim = len(filter_sizes) * filter_num
        self.fc = nn.Linear(self.last_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        '''权重初始化'''
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
    
    def forward(self, data):
        '''前向传播到输出层之前'''
        x = data.x # 只取text部分的特征
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        return x
    
    def out(self, data):
        '''前向传播到输出层'''
        x = self.forward(data)
        x = F.dropout(x, training=self.training)
        outputs = self.fc(x)
        return outputs

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(BiLSTM, self).__init__()
        emb_dim = 256
        self.n_hidden = 150
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, self.n_hidden, bidirectional=True)
        self.last_dim = self.n_hidden * 2
        self.fc = nn.Linear(self.last_dim, num_classes)

        self.init_weights()
    
    def init_weights(self):
        '''权重初始化'''
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, data):
        '''前向传播到输出层之前'''
        x = data.x # 只取text部分的特征
        x = self.embedding(x) # input : [batch_size, len_seq, embedding_dim]
        x = x.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]
        # print(x.shape)
        hidden_state = torch.zeros(1*2, x.shape[1], self.n_hidden, requires_grad=True).to(x.device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, x.shape[1], self.n_hidden, requires_grad=True).to(x.device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
        final_hidden_state = final_hidden_state.permute(1, 0, 2).reshape(-1, self.last_dim)
        return final_hidden_state 
    
    def out(self, data):
        '''前向传播到输出层'''
        x = self.forward(data)
        x = F.dropout(x, training=self.training)
        outputs = self.fc(x)
        return outputs

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(TransformerModel, self).__init__()
        ntokens = vocab_size # the size of vocabulary
        ninp = 256 # embedding dimension
        nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 1 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 1 # the number of heads in the multiheadattention models
        dropout = 0.5 # the dropout value
        self.src_mask = None
        self.embedding = nn.Embedding(ntokens, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.fc = nn.Linear(ninp, num_classes)
        self.ninp = ninp
        self.last_dim = ninp
        
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        '''权重初始化'''
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, data):
        '''前向传播到输出层之前'''
        src = data.x # 只取text部分的特征
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = torch.mean(output, dim=1)
        return output
    
    def out(self, data):
        '''前向传播到输出层'''
        x = self.forward(data)
        x = F.dropout(x, training=self.training)
        outputs = self.fc(x)
        return outputs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)