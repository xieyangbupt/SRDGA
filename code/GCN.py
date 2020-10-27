import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN(nn.Module):
    '''
    GCN的实现
    '''
    def __init__(self, encoder, gcn, num_classes):
        super(GCN, self).__init__()
        self.encoder = encoder
        d = self.encoder.last_dim
        dim = self.encoder.last_dim
        # dim = 100
        if gcn == 'GCN':
            self.conv1 = GCNConv(d, dim, flow='source_to_target')
            self.conv2 = GCNConv(dim, dim, flow='source_to_target')
        elif gcn == 'SAGE':
            self.conv1 = SAGEConv(d, dim, flow='source_to_target')
            self.conv2 = SAGEConv(dim, dim, flow='source_to_target')
        elif gcn == 'GAT':
            self.conv1 = GATConv(d, dim, flow='source_to_target')
            self.conv2 = GATConv(dim, dim, flow='source_to_target')
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, data):
        '''前向传播到输出层之前'''
        # encoder
        x = self.encoder(data)
        # GCN
        conv1_in = F.dropout(x, training=self.training)
        conv1_out = F.relu(self.conv1(conv1_in, data.edge_index))
        conv2_in = F.dropout(conv1_out, training=self.training)
        conv2_out = F.relu(self.conv2(conv2_in, data.edge_index))
        return conv2_out
    
    def out(self, data):
        '''前向传播到输出层'''
        x = self.forward(data)
        x = F.dropout(x, training=self.training)
        outputs = self.fc(x)
        return outputs

class DGGCN(nn.Module):
    '''
    DGGCN的实现
    '''
    def __init__(self, encoder, gcn, num_classes):
        super(DGGCN, self).__init__()
        self.encoder = encoder
        d = self.encoder.last_dim
        dim = self.encoder.last_dim
        # dim = 100
        if gcn == 'DGGCN':
            self.conv1 = GCNConv(d, dim, flow='source_to_target')
            self.conv2 = GCNConv(dim, dim, flow='source_to_target')
        elif gcn == 'DGSAGE':
            self.conv1 = SAGEConv(d, dim, flow='source_to_target')
            self.conv2 = SAGEConv(dim, dim, flow='source_to_target')
        elif gcn == 'DGGAT':
            self.conv1 = GATConv(d, dim, flow='source_to_target')
            self.conv2 = GATConv(dim, dim, flow='source_to_target')
        # 门控
        self.w11 = nn.Parameter(torch.Tensor(dim, dim))
        self.w12 = nn.Parameter(torch.Tensor(dim, dim))
        self.b1 = nn.Parameter(torch.Tensor(dim))
        self.w21 = nn.Parameter(torch.Tensor(dim, dim))
        self.w22 = nn.Parameter(torch.Tensor(dim, dim))
        self.b2 = nn.Parameter(torch.Tensor(dim))

        self.fc = nn.Linear(dim, num_classes)
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.w11, a=math.sqrt(5))
        init.kaiming_normal_(self.w12, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w11)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b1, -bound, bound)

        init.kaiming_normal_(self.w21, a=math.sqrt(5))
        init.kaiming_normal_(self.w22, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w21)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b2, -bound, bound)
    
    def forward(self, data):
        '''前向传播到输出层之前'''
        # encoder
        x = self.encoder(data)
        # GCN layer1
        conv1_in = F.dropout(x, training=self.training)
        conv1_out1 = F.relu(self.conv1(conv1_in, data.edge_index))
        conv1_out2 = F.relu(self.conv1(conv1_in, data.edge_index_reverse))
        # Gate
        G = torch.sigmoid(F.linear(conv1_out1, self.w11) + F.linear(conv1_out2, self.w12) + self.b1)
        conv1_out = G*conv1_out1 + (1-G)*conv1_out2
        # GCN layer2
        conv2_in = F.dropout(conv1_out, training=self.training)
        conv2_out1 = F.relu(self.conv2(conv2_in, data.edge_index))
        conv2_out2 = F.relu(self.conv2(conv2_in, data.edge_index_reverse))
        # Gate
        G = torch.sigmoid(F.linear(conv2_out1, self.w21) + F.linear(conv2_out2, self.w22) + self.b2)
        conv2_out = G*conv2_out1 + (1-G)*conv2_out2
        return conv2_out
    
    def out(self, data):
        '''前向传播到输出层'''
        x = self.forward(data)
        x = F.dropout(x, training=self.training)
        outputs = self.fc(x)
        return outputs