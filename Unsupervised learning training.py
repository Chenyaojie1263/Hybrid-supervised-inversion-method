import einops
import scipy.signal
from einops import rearrange
import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, BatchNorm2d, Upsample
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import DataLoader
from DATA_un import MyDataSet_un
from Unet_GRU_MODEL_Unsupervised import Unet_GRU_Un
import numpy as np
import scipy.io as scio
import scipy.io as sio
import datetime
import os


if __name__ == '__main__':
    epoch_num = 1000
    batch_size_train = 4
    train_num = 10000
    ite_num = 0

    A = scio.loadmat("./xxx")['A']  #Enter forward operator A filename
    A = torch.tensor(A,dtype=torch.float32)
    A = A.cuda()

    data = MyDataSet_un()
    data_loader = DataLoader(data, batch_size=batch_size_train, shuffle=True)

    loss_fn2 = torch.nn.MSELoss()
    loss_fn1 = torch.nn.L1Loss()
    model = Unet_GRU_Un()

    model = model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    Loss_un = []
    Iteration = []
    Loss_eval = []
    Iteration_eval = []

    for epoch in range(0, epoch_num):
        model.train()
        for index, data in enumerate(data_loader):
            ite_num = ite_num + 1
            seq_data, seis, L0 = data

            seq_data, seis, L0 = seq_data.cuda(), seis.cuda(), L0.cuda()
            prd_tag = model(seq_data)

            prd_tag_loss = prd_tag.permute(0, 2, 1)
            seis_loss = seis
            L0_loss = L0
            A_loss = A
            seis_loss1 = torch.matmul(A_loss, prd_tag_loss)
            loss = loss_fn1(seis_loss1, seis_loss) + 0.1 * loss_fn1(prd_tag_loss, L0_loss)


            opt.zero_grad()
            loss.backward()
            opt.step()
            Iteration.append(ite_num)

        if epoch == 300:
            torch.save(model.state_dict(), "model_un.pth")
            print("Save successfully")
            exit()