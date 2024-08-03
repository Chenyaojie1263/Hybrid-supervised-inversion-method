import einops
import scipy.signal
from einops import rearrange
import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, BatchNorm2d, Upsample
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import DataLoader
from Unet_GRU_MODEL import Unet_GRU
from DATA import MyDataSet
import numpy as np
import scipy.io as scio
import scipy.io as sio
import datetime
import os
if __name__ == '__main__':
    epoch_num = 1000
    batch_size_train = 4
    train_num = 1000
    ite_num = 0
    data = MyDataSet()
    data_loader = DataLoader(data, batch_size=batch_size_train, shuffle=True)
    loss_fn1 = torch.nn.MSELoss()
    model = Unet_GRU()
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    Loss = []
    Iteration = []

    for epoch in range(0, epoch_num):
        model.train()
        for index, data in enumerate(data_loader):
            ite_num = ite_num + 1
            seq_data, label2, L0 = data
            seq_data, label2, L0 = seq_data.cuda(), label2.cuda(), L0.cuda()
            prd_tag = model(seq_data , L0)
            loss = loss_fn1(prd_tag, label2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            Iteration.append(ite_num)

        if epoch == 300:
            torch.save(model.state_dict(), "model1.pth")
            print("Save successfully")
            exit()