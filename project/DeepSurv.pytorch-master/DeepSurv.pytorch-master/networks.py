from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order                   #Lasso的order为1,Ridge的order为2,即正则化范数
        self.weight_decay = weight_decay     #权重衰减率lamda

    def __call__(self, model:torch.nn.Module):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0                               #正则项loss
        for name, w in model.named_parameters(): 
            if 'weight' in name:                       #只更新权重
                reg_loss += torch.norm(w, p=self.order)    # 循环结束求得||w||2
        reg_loss = self.weight_decay * reg_loss            #  正则化loss = lambda*||w||2
        return reg_loss

class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # 添加dropout层，注意不能在第一层加，且config内要求有drop参数
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1])) #设置全连接层，连接第i层与第i+1层
            if self.norm: # adds batchnormalize layer       
                layers.append(nn.BatchNorm1d(self.dims[i+1]))      #如果config要求保持输入分布的不变性，则调用这句话
            # adds activation layer                                #调用激活函数
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class NegativeLogLikelihood(nn.Module):    #loss函数
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg) 

    def forward(self, risk_pred, y, e, model):      #risk_pred是神经网络得出的h(x1),h(x2),...,h(xk)
        mask = torch.ones(y.shape[0], y.shape[0])   #建立一个所有元素为1的mask方阵
        mask[(y.T - y) > 0] = 0                     #基准事件大于存活大于其它对比事件的无需考虑,行-列
        log_loss = torch.exp(risk_pred) * mask      
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)   
        log_loss = torch.log(log_loss).reshape(-1,1)              #整理成列数组
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)                  #调用Regularization的call函数，计算正则化loss
        return neg_log_loss + l2_loss