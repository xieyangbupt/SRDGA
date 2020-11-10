import os.path as osp
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from visdom import Visdom
from TrainTest import train, test, pred_unlabeled
from Dataset import Weibo
from Encoder import BiLSTM, TextCNN, FastText, TransformerModel
from GCN import GCN, SRDGAGCN

def main(encoder, gcn, edge):
    '''main函数，节点分类'''
    # 定义超参数
    parser = argparse.ArgumentParser(description='Hyperparameters of the program')
    # 数据集参数
    parser.add_argument('--dataset-path', type=str, default=osp.join('data', 'weibo-1'),
                        help='the path of dataset')
    parser.add_argument('--seg', type=str, default='seg',
                        help='how the word is split')
    parser.add_argument('--edge_type', type=str, default=edge,
                        help='the edge type (social.origin/undirected/reverse)')
    # 模型参数
    parser.add_argument('--encoder', type=str, default=encoder,
                        help='the model that used to encode the text')
    parser.add_argument('--gcn', type=str, default=gcn,
                        help='the model that used to perform the graph covolution')
    # 优化器参数
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate of model')
    parser.add_argument('--wd', type=float, default=5e-3,
                        help='weight decay of model')
    # 训练测试参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--no-cuda', type=bool, default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', type=bool, default=False,
                        help='For saving the current Model')
    # visdom参数
    parser.add_argument('-port', type=int, default=8097,
                        help='port the visdom server is running on.')
    parser.add_argument('-server', type=str, default='http://localhost',
                        help='Server address of the target to run the visdom on.')

    # 所有参数
    args = parser.parse_args()
    print(args)
    # 运算设备
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Trainning on', device)
    # seed相同保证tensor随机初始化的值是一样的
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    # 载入数据集，不用DataLoader
    dataset = Weibo(root=args.dataset_path, seg=args.seg, edge_type=args.edge_type)
    data = dataset[0]
            
    vocab_size = dataset.vocab_size # 训练集词典
    num_classes = dataset.num_classes # 分类个数
    print('vocab size:', vocab_size)
    print('node num:', data.num_nodes)
    print('edge num:', data.num_edges)
    print('classes num:', num_classes)
    
    # Encoder
    if args.encoder == 'TextCNN':
        encoder = TextCNN(vocab_size, num_classes)
    elif args.encoder == 'BiLSTM':
        encoder = BiLSTM(vocab_size, num_classes)
    elif args.encoder == 'FastText':
        encoder = FastText(vocab_size, num_classes)
    elif args.encoder == 'TransformerModel':
        encoder = TransformerModel(vocab_size, num_classes)
    
    # 模型实例
    if args.gcn:
        if args.gcn in ['SRDGAGAT', 'SRDGAGCN', 'SRDGASAGE']:
            model = SRDGAGCN(encoder, args.gcn, num_classes).to(device)
        elif args.gcn in ['GAT', 'GCN', 'SAGE']:
            model = GCN(encoder, args.gcn, num_classes).to(device)
    else:
        model = encoder.to(device)
    
    # 打印模型参数量模型和名称
    print('parameters:', sum(param.numel() for param in model.parameters()))
    print(model)

    # 利用visdom进行可视化
    env_name = '{}.{}'.format(args.encoder, args.seg)
    if args.gcn: # 使用GCN
        env_name += '.{}.{}'.format(args.gcn, args.edge_type)
    print(env_name)
    viz = Visdom(env=env_name, port=args.port, server=args.server)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # 学习速率调整器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 迭代训练
    max_test_acc = 0
    best_test_pred = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        # 训练
        train_loss, train_acc = train(model, device, data, optimizer, criterion)
        # 可视化train_loss
        viz.line(X=torch.FloatTensor([epoch]),
                Y=torch.FloatTensor([train_loss]),
                win='train_loss',
                update='append' if epoch > 1 else None,
                opts={'title':'train loss', 'xlabel':'epoch'})
        # 可视化train_acc
        viz.line(X=torch.FloatTensor([epoch]), 
                Y=torch.FloatTensor([train_acc]), 
                win='train_accuracy',
                update='append' if epoch > 1 else None,
                opts={'title':'train accuracy', 'xlabel':'epoch'})
        # 测试
        test_loss, test_acc, test_pred = test(model, device, data, criterion)
        # 记录测试最大值
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            best_test_pred = test_pred
            unlabeled_pred = pred_unlabeled(model, device, data)
        # 更新学习率
        # scheduler.step()
        # 可视化test_loss
        viz.line(X=torch.FloatTensor([epoch]), 
                Y=torch.FloatTensor([test_loss]), 
                win='test_loss',
                update='append' if epoch > 1 else None,
                opts={'title':'test loss', 'xlabel':'epoch'})
        # 可视化test_acc
        viz.line(X=torch.FloatTensor([epoch]), 
                Y=torch.FloatTensor([test_acc]), 
                win='test_accuracy',
                update='append' if epoch > 1 else None,
                opts={'title':'test accuracy', 'xlabel':'epoch'})
        # 打印相关信息
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        print('Epoch: %d' %(epoch), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
    # 保存模型
    if (args.save_model):
        torch.save(model.state_dict(), args.model+'.pt')
    # 保存最好的test结果
    pred_path = osp.join(args.dataset_path, 'pred', 'test', env_name)
    with open(pred_path, 'w', encoding='utf8') as f:
        f.write('max test acc: %f\n' % max_test_acc)
        print(*best_test_pred, sep='\n', file=f)
    # 保存unlabeled上的预测结果
    pred_path = osp.join(args.dataset_path, 'pred', 'unlabeled', env_name)
    with open(pred_path, 'w', encoding='utf8') as f:
        f.write('max test acc: %f\n' % max_test_acc)
        print(*unlabeled_pred, sep='\n', file=f)
    # 画出节点嵌入
    # embedding_cluster(model, dataset, device)
    print(model)

# 程序入口
if __name__ == '__main__':
    for e in ['FastText', 'TextCNN', 'BiLSTM', 'TransformerModel']:
        for g in ['GCN']:
            for edge in ['social.origin']:
                main(e, g, edge)