from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt
from torch.utils.data import DataLoader

from networks import DeepSurv
from networks import NegativeLogLikelihood
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger

def train(ini_file):
    ''' Performs training according to .ini file
    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(config['network']).to(device='cpu')
    criterion = NegativeLogLikelihood(config['network']).to(device='cpu')
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(model.parameters(),lr = config['train']['learning_rate'])
    # constructs data loaders based on configuration
    train_dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)
    test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=train_dataset.__len__()) #不进行分批训练,batch_size = X.shape[0]
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())    #不进行分批训练,batch_size = X.shape[0]
    # training
    best_c_index = 0
    flag = 0
    for epoch in range(1, config['train']['epochs']+1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
        # train step
        model.train()
        for X, y, e in train_loader:   #一个batch会直接传入循环！！！
            # makes predictions
            risk_pred = model(X)   #计算预测值
            train_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)
            # updates parameters
            optimizer.zero_grad()  #初始化梯度为0
            train_loss.backward()  #反向传播梯度
            optimizer.step()       #更新model中所有参数
        # valid step
        model.eval()
        for X, y, e in test_loader:
            # makes predictions
            with torch.no_grad():           #测试过程不需要更新梯度了
                risk_pred = model(X)        #求解预测h(x)
                valid_loss = criterion(risk_pred, y, e, model)  #测试loss
                valid_c = c_index(-risk_pred, y, e)             #测试的C-index
                if best_c_index < valid_c:       #更新最优的C-index
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                else:                         #若无法更新，更新等待次数
                    flag += 1                 #超过耐心值次数仍无法更新，直接返回C-index
                    if flag >= patience:
                        return best_c_index
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}(test:{:.8f})\tc-index: {:.8f}(test:{:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
    return best_c_index

if __name__ == '__main__':
    # global settings
    logs_dir = "project\\DeepSurv.pytorch-master\\DeepSurv.pytorch-master"
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logger = create_logger(os.path.join(logs_dir,'logs'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs_dir = "project\\DeepSurv.pytorch-master\\DeepSurv.pytorch-master\\configs"
    params = [
        #('Simulated Linear', 'linear.ini'),
        #('Simulated Nonlinear', 'gaussian.ini'),
        #('WHAS', 'whas.ini'),
        #('SUPPORT', 'support.ini'),
        ('METABRIC', 'metabric.ini'),
        #('Simulated Treatment', 'treatment.ini'),
        #('Rotterdam & GBSG', 'gbsg.ini')
            ]
    patience = 50  #等待次数
    # training
    headers = []
    values = []
    for name, ini_file in params:
        logger.info(f'Running {name}({ini_file})...')
        best_c_index = train(os.path.join(configs_dir, ini_file))
        headers.append(name)
        values.append('{:.6f}'.format(best_c_index))
        print('')
        logger.info("The best valid c-index: {}".format(best_c_index))
        logger.info('')
    # prints results
    tb = pt.PrettyTable()
    tb.field_names = headers
    tb.add_row(values)
    logger.info(tb)