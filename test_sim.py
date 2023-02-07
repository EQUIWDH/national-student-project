import numpy as np
from generate_surv_data import SimulatedData
from abess import CoxPHSurvivalAnalysis

import os
import torch
import torch.optim as optim
import prettytable as pt
from torch.utils.data import DataLoader

from networks import DeepSurv
from networks import NegativeLogLikelihood
from datasets import SurvivalDataset
import h5py
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger
from time import time
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

sim_generate = SimulatedData(hr_ratio=1.0, average_death = 5, censor_mode = 'observed_p',observed_p = 0.7,num_features = 10000, num_var = 20)
sim_com = sim_generate.generate_data(N = 800, method='comb')
sim_x = sim_com['x']
sim_y = np.c_[sim_com['t'],sim_com['e']]
x_train,x_test,y_train,y_test = train_test_split(sim_x,sim_y,test_size=0.2, random_state=1)
x_test,x_validation,y_test,y_validation = train_test_split(x_test,y_test,test_size=0.5,random_state=1)
d_train = y_train[:,0]
e_train = y_train[:,1]
d_test = y_test[:,0]
e_test = y_test[:,1]
d_validation = y_test[:,0]
e_validation = y_validation[:,1]

t_1 = time()
abess = CoxPHSurvivalAnalysis(ic_type = 'aic')
t_1 = time() - t_1

abess.fit(sim_x,sim_y)
X_train = x_train[:,np.squeeze(np.array(np.where(abess.coef_ != 0)))]
X_test = x_test[:,np.squeeze(np.array(np.where(abess.coef_ != 0)))]
X_validation = x_validation[:,np.squeeze(np.array(np.where(abess.coef_ != 0)))]


with h5py.File("combination.h5", 'w') as f:
    test = f.create_group("test") # 在根目录下创建gruop文件夹:dogs
    test.create_dataset('e',data = e_test) # 根目录下有一个含5张猫图片的dataset文件
    test.create_dataset('t',data = d_test)
    test.create_dataset('x',data = X_test)
    train = f.create_group("train") # 在根目录下创建gruop文件夹:dogs
    train.create_dataset('e',data = e_train) # 根目录下有一个含5张猫图片的dataset文件
    train.create_dataset('t',data = d_train)
    train.create_dataset('x',data = X_train)

with h5py.File("whole.h5", 'w') as f:
    test = f.create_group("test") # 在根目录下创建gruop文件夹:dogs
    test.create_dataset('e',data = e_test) # 根目录下有一个含5张猫图片的dataset文件
    test.create_dataset('t',data = d_test)
    test.create_dataset('x',data = x_test)
    train = f.create_group("train") # 在根目录下创建gruop文件夹:dogs
    train.create_dataset('e',data = e_train) # 根目录下有一个含5张猫图片的dataset文件
    train.create_dataset('t',data = d_train)
    train.create_dataset('x',data = x_train)

with h5py.File("validation.h5",'w') as f:
    combine = f.create_group("combine") # 在根目录下创建gruop文件夹:dogs
    combine.create_dataset('e',data = e_validation) # 根目录下有一个含5张猫图片的dataset文件
    combine.create_dataset('t',data = d_validation)
    combine.create_dataset('x',data = X_validation)
    whole = f.create_group("whole") # 在根目录下创建gruop文件夹:dogs
    whole.create_dataset('e',data = e_validation) # 根目录下有一个含5张猫图片的dataset文件
    whole.create_dataset('t',data = d_validation)
    whole.create_dataset('x',data = x_validation)


times = []
hyper_grid = [] 
model_names = ['deep_abess.pth','only_deep.pth']
data_files = ["combination.h5" , "whole.h5"]
hyper_names = ['deep_abess_hyper.pkl','only_depp.pkl']
hyper_grid.append({'drop' : list(np.linspace(0.0, 0.5,100)), 'norm' :[True], 'dims': [[len(X_train.T), 18, 18, 1], [len(X_train.T), 8, 8,8, 1], [len(X_train.T), 16, 8, 1], [len(X_train.T), 8, 16, 1]], 'activation' :['SELU'],'l2_reg' : [0], 'epochs' : [1000], 'learning_rate' : list(10**np.linspace(-7,-3,100)), 'lr_decay_rate' : list(np.linspace(0,0.001,100)), 'optimizer' : ['Adam'], 'patience': [100]})
hyper_grid.append({'drop' : list(np.linspace(0.0, 0.5,100)), 'norm' :[True], 'dims': [[len(x_train.T), 18, 18, 1], [len(x_train.T), 8, 8,8, 1], [len(x_train.T), 16, 8, 1], [len(x_train.T), 8, 16, 1]], 'activation' :['SELU'],'l2_reg' : [0], 'epochs' : [1000], 'learning_rate' : list(10**np.linspace(-7,-3,100)), 'lr_decay_rate' : list(np.linspace(0,0.001,100)), 'optimizer' : ['Adam'], 'patience': [100]})

def train(hyper_para,data_file):
    ''' Performs training according to .ini file
    
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(hyper_para).to(device='cpu')
    criterion = NegativeLogLikelihood(hyper_para).to(device='cpu')
    optimizer = eval('optim.{}'.format(hyper_para['optimizer']))(model.parameters(),lr = hyper_para['learning_rate'])
    # constructs data loaders based on configuration
    train_dataset = SurvivalDataset(data_file, is_train=True)
    test_dataset = SurvivalDataset(data_file, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=train_dataset.__len__()) #不进行分批训练,batch_size = X.shape[0]
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())    #不进行分批训练,batch_size = X.shape[0]
    # training
    best_c_index = 0
    flag = 0
    for epoch in range(1, hyper_para['epochs']+1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  hyper_para['learning_rate'],
                                  hyper_para['lr_decay_rate'])
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
                else:                         #若无法更新，更新等待次数
                    flag += 1                 #超过耐心值次数仍无法更新，直接返回C-index
                    if flag >= hyper_para['patience'] :
                        print('\n')
                        return best_c_index, model
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}(validation:{:.8f})\tc-index: {:.8f}(validation:{:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
    print('\n')
    return best_c_index, model

if __name__ == '__main__':
# global settings
    logs_dir = ""
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    hypers_dir = os.path.join(logs_dir, 'hypers')
    if not os.path.exists(hypers_dir):
        os.makedirs(hypers_dir)
    logger = create_logger(os.path.join(logs_dir,'logs'))
    device = torch.device('cpu')
    params = [
          'combination',
          'whole'
            ]
    # training
    headers = []
    values = []
    random_searching = 1
    for ind,name in enumerate(params):
        logger.info(f'Running the hyper para searching of {name}.h5 dataset...')
        best_c_index = 0
        best_hyper = {}
        t_2 = time()
        for i in range(random_searching):
            np.random.seed(i)
            hyper_para = {k: hyper_grid[ind][k][np.random.randint(len(hyper_grid[ind][k]))] for k, v in hyper_grid[ind].items()}
            c_ind, model = train(hyper_para,data_files[ind]) #返回validation_test_c_index
            if c_ind > best_c_index:
                best_c_index = c_ind
                best_hyper = hyper_para
                #保存模型和超参
                torch.save(model, os.path.join(models_dir, model_names[ind]))
                with open(os.path.join(hypers_dir, hyper_names[ind]), 'wb') as file:
                    pickle.dump(best_hyper, file, 4)
        t_2 = time() - t_2
        print(f'=> end hyper hyper searching of {name}.h5 dataset...')
        times.append(t_2)
        headers.append(name)
        values.append('{:.6f}'.format(best_c_index))
        print('')
        logger.info("The best valid c-index: {}".format(best_c_index))
        logger.info('')
    # prints results
    times[0] = times[0] + t_1
    tb = pt.PrettyTable()
    tb.field_names = headers
    tb.add_row(values)
    logger.info(tb)
    print(f'Deepsurv-ABESS running time is {times[0]}s, Deepsurv running time is {times[1]}s')
    '''
    plt.figure(dpi = 100)
    plt.title('loss on validation')
    plt.plot(range(1,len(loss[0])+1),loss[0],label = 'ABESS-Deepsurv')
    plt.plot(range(1,len(loss[1])+1),loss[1],label = 'Deepsurv')
    plt.xlabel('training epoch')
    plt.ylabel('loss on validation')
    plt.legend()
    plt.show()
    '''
