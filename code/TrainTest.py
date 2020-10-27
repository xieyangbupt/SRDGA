import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visdom import Visdom

def train(model, device, data, optimizer, criterion):
    '''训练函数'''
    # 模型设置成训练模式
    model.train()
    # 数据转化成对应device上
    data = data.to(device)
    target = data.y[data.labeled_mask]
    # 模型梯度清零
    optimizer.zero_grad()
    # 得到模型输出结果
    output = model.out(data)
    out = output[data.labeled_mask]
    # 计算loss
    loss = criterion(out, target)
    train_loss = loss.item()
    # 计算accuracy
    train_acc = (out.argmax(1) == target).sum().item()
    # 梯度反向传播
    loss.backward()
    # 更新模型参数
    optimizer.step()
    # 返回平均在每个样本上的loss和acc
    return train_loss/len(torch.nonzero(data.labeled_mask)), train_acc/len(torch.nonzero(data.labeled_mask))
    
def test(model, device, data, criterion):
    '''测试函数'''
    # 模型设置成预测模式
    model.eval()
    # 模型在预测模式下不需要追踪梯度
    with torch.no_grad():
        data = data.to(device)
        target = data.y[data.test_mask]
        # 得到模型输出结果
        output = model.out(data)
        out = output[data.test_mask]
        # 计算loss
        loss = criterion(out, target)
        loss = loss.item()
        # 计算accuracy
        pred = out.argmax(1)
        acc = (pred == target).sum().item()
        # 返回平均在每个样本上的loss，acc和pred
        return loss/len(torch.nonzero(data.test_mask)), acc/len(torch.nonzero(data.test_mask)), pred.tolist()

def pred_unlabeled(model, device, data):
    '''模型预测unlabeled'''
    # 模型设置成预测模式
    model.eval()
    # 模型在预测模式下不需要追踪梯度
    with torch.no_grad():
        data = data.to(device)
        # 得到模型输出结果
        output = model.out(data)
        out = output[data.unlabeled_mask]
        # 计算pred
        pred = out.argmax(1)
        # 返回pred
        return pred.tolist()
