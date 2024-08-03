import einops
import scipy.signal
from einops import rearrange
import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, BatchNorm2d, Upsample
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import DataLoader
from DATA import MyDataSet
import numpy as np
import scipy.io as scio
import scipy.io as sio
import datetime
import os

seq_len = 2
hidden_size = 32
number_layer = 2
Input_size = 1461
num_directions = 1
class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Unet_GRU(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Unet_GRU, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Conv_2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=2, bias=True)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=0)
        self.linear = nn.Linear(1458,1464)

        self.GRU = nn.GRU(input_size=Input_size, hidden_size=hidden_size,
                          num_layers=number_layer, batch_first=True, dropout=0.1)
        self.out = nn.Sequential(nn.Linear(hidden_size, 1464))

    def forward(self, x,L0):
        x1 = self.Conv_2(x)

        e1 = self.Conv1(x1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        d3 = self.Up3(e3)
        crop_size = (e2.shape[2] - d3.shape[2], e2.shape[3] - d3.shape[3])
        e2_cropped = e2[:, :, crop_size[0]:, crop_size[1]:]

        d3 = torch.cat((e2_cropped, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)

        crop_size = (e1.shape[2] - d2.shape[2], e1.shape[3] - d2.shape[3])
        e1_cropped = e1[:, :, crop_size[0]:, crop_size[1]:]
        d2 = torch.cat((e1_cropped, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)
        x_unet = self.linear(out)
        n, c, h, w = x.shape
        x_GRU = x.reshape(n, c * h, w)
        x_GRU, _ = self.GRU(x_GRU)
        x_GRU = self.out(x_GRU)
        n1, c1, h1, w1 = x_unet.shape
        x_unet = x_unet.reshape(n1, c1 * h1, w1)
        x1 = (x_unet+x_GRU)/2
        Result = (x1 + L0) / 2
        return Result