import os.path as osp
import pickle
import torch

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

def pickle_load(var_path, dtype):
    '''使用pickle从文件中读取变量'''
    print('pickling from', var_path)
    with open(var_path, 'rb') as f:
        var = pickle.load(f)
    return torch.tensor(var, dtype=dtype)

def sample_mask(index, num_nodes):
    '''采样mask'''
    mask = torch.zeros((num_nodes, ), dtype=torch.bool)
    mask[index] = 1
    return mask

class Weibo(InMemoryDataset):
    '''
    微博数据集
    '''
    def __init__(self, root, seg, edge_type, transform=None, pre_transform=None):
        self.seg = seg
        self.edge_type = edge_type
        super(Weibo, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def vocab_size(self):
        '''词典大小'''
        with open(osp.join(self.raw_dir, 'vocab.{}'.format(self.seg)), 'rb') as fin:
            vocab = pickle.load(fin)
        return len(vocab)
    
    @property
    def raw_file_names(self):
        names = ['xl.{}'.format(self.seg), 'xu.{}'.format(self.seg), 'xt.{}'.format(self.seg)]
        names.extend(['yl', 'yu', 'yt'])
        names.extend(['edge.{}'.format(self.edge_type)])
        return names

    @property
    def processed_file_names(self):
        processed_file = 'data.{}.{}.pt'.format(self.edge_type, self.seg)
        return processed_file

    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def read_reverse_edge(self, folder):
        '''读取逆向边'''
        return pickle_load(var_path=osp.join(folder, 'edge.social.reverse'), dtype=torch.long)
    
    def read_weibo_data(self, folder):
        '''读取数据'''
        names = self.raw_file_names # 获取raw中的文件名
        items = [pickle_load(var_path=osp.join(folder, name), dtype=torch.long) for name in names] # pickle读数据
        lx, ux, tx, ly, uy, ty, edge_index = items
        # 构建标记数据，未标数据，测试数据下标
        labeled_index = torch.arange(lx.size(0), dtype=torch.long) # labeled index
        unlabeled_index = torch.arange(lx.size(0), lx.size(0)+ux.size(0), dtype=torch.long) # unlabeled index
        test_index = torch.arange(lx.size(0)+ux.size(0), lx.size(0)+ux.size(0)+tx.size(0), dtype=torch.long) # test index
        # 拼接数据
        x = torch.cat([lx, ux, tx], dim=0)
        y = torch.cat([ly, uy, ty], dim=0)
        # 构建训练和测试的mask
        labeled_mask = sample_mask(labeled_index, num_nodes=x.size(0))
        train_mask = sample_mask(labeled_index, num_nodes=x.size(0))
        unlabeled_mask = sample_mask(unlabeled_index, num_nodes=x.size(0))
        test_mask = sample_mask(test_index, num_nodes=x.size(0))
        # 构造data
        data = Data(x=x, edge_index=edge_index, y=y)
        data.edge_index_reverse = self.read_reverse_edge(folder) # 逆向社交边
        # mask
        data.labeled_mask = labeled_mask
        data.unlabeled_mask = unlabeled_mask
        data.train_mask = train_mask
        data.test_mask = test_mask
        return data

    def process(self):
        # Read data into huge `Data` list.
        data = self.read_weibo_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    