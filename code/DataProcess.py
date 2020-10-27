import os.path as osp
import pickle
import numpy as np
from functools import partial

def pickle_dump(var, path):
    '''pickle存储变量'''
    print('pickling', path)
    with open(path, 'wb') as f:
        pickle.dump(var, f)

class WeiboProcess:
    '''
    微博数据集预处理
    '''
    def __init__(self, root, seg, sen_len):
        self.root = root
        # 标注数据、未标注数据和测试数据的已经分好词的路径
        self.labeled_path = osp.join(self.root, seg, 'labeled.csv')
        self.unlabeled_path = osp.join(self.root, seg, 'unlabeled.csv')
        self.test_path = osp.join(self.root, seg, 'test.csv')
        # 预处理之后的路径
        raw_path = osp.join(self.root, 'raw')
        self.lx_path = osp.join(raw_path, 'xl.{}'.format(seg))
        self.tx_path = osp.join(raw_path, 'xt.{}'.format(seg))
        self.ux_path = osp.join(raw_path, 'xu.{}'.format(seg))
        self.ly_path = osp.join(raw_path, 'yl')
        self.ty_path = osp.join(raw_path, 'yt')
        self.uy_path = osp.join(raw_path, 'yu')
        self.vocab_path = osp.join(raw_path, 'vocab.{}'.format(seg))
        # 句子长度
        self.sen_len = sen_len
        # 词典
        self.vocab = {}
        self.vocab['<PAD>'] = 0 # padding的时候添加<PAD>
        # self.vocab['<UNK>'] = 1 # 测试数据中的OOV用<UNK>表示
        # 标签映射
        self.LABEL_MAP = {'-1':0, '0':1, '1':2}
    
    def process(self):
        '''将原始数据处理成pickle数据'''
        self.process_xy(self.labeled_path, self.lx_path, self.ly_path, train=True) # 处理labeled数据
        self.process_xy(self.test_path, self.tx_path, self.ty_path, train=True) # 处理test数据
        self.process_xy(self.unlabeled_path, self.ux_path, self.uy_path, train=True) # 处理unlabeled数据
        pickle_dump(self.vocab, self.vocab_path) # 保存词典

    def process_xy(self, data_path, x_path, y_path, train):
        '''将文本转化成下标，作为节点文本特征'''
        with open(data_path, 'r', encoding='utf8') as f:
            data = [line.split('\t') for line in f]
            text = [d[2] for d in data] # 文本
            text_idx = list(map(partial(self.text_to_idx, train=train), text)) # 文本转化成下标
            x = text_idx
            if 'unlabeled' in data_path: # unlabeled的标签全为0
                label_norm = [0 for _ in range(len(x))]
            else: # labeled、test、noatti的标签都处理成norm
                label = [d[-1].strip() for d in data] # 标签
                label_norm = list(map(lambda y:self.LABEL_MAP[y], label)) # 标签转化成从0开始的下标
            y = label_norm
            # pickle存储
            pickle_dump(x, x_path)
            pickle_dump(y, y_path)

    def text_to_idx(self, seq, train):
        '''将文字序列转化成下标，同时进行截断或补零'''
        # 如果文本为空，返回全部为<PAD>的下标
        seq = seq.strip()
        if seq == '': return [self.vocab['<PAD>'] for _ in range(self.sen_len)]
        # 否则按空格分词
        else: seq = seq.split(' ')
        # 将训练集单词加入词典
        if train:
            for word in seq:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        idxs = []
        for i in range(self.sen_len):
            if i < len(seq):
                if seq[i] in self.vocab:
                    idxs.append(self.vocab[seq[i]])
                else:
                    idxs.append(self.vocab['<UNK>'])
            else: # 在结尾补零
                idxs.append(self.vocab['<PAD>'])
        return idxs

class EdgeProcess:
    '''
    边的预处理
    '''
    def __init__(self, root, edge_type):
        self.root = root
        self.edge_path = osp.join(self.root, 'edge', 'edge.{}.csv'.format(edge_type))
        self.edge_index_path = osp.join(self.root, 'raw', 'edge.{}'.format(edge_type))
    
    def process(self):
        '''处理边'''
        # 读取边数据
        edge_index = [[], []]
        with open(self.edge_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                source, target = int(line[0]), int(line[1])
                edge_index[0].append(source)
                edge_index[1].append(target)
        # pickle存储
        pickle_dump(edge_index, self.edge_index_path)

if __name__ == "__main__":
    from setting import data_dir, LEN
    print(data_dir)
    seg = 'seg'
    sen_len = LEN[seg]
    data_process = WeiboProcess(data_dir, seg, sen_len)
    data_process.process()

    edge_type = 'social.origin' # social.origin/undirected/reverse
    data_process = EdgeProcess(data_dir, edge_type)
    data_process.process()
