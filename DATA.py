import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio
import os


class MyDataSet(Dataset):
    def __init__(self):
        self.root_dir = "./xxx"  # Input data file name
        self.seis_dir = "xxx"  # Input seismic data file name
        self.seis_path1 = os.path.join(self.root_dir, self.seis_dir)
        self.seis_path = os.listdir(self.seis_path1)

        self.label_dir = "L_all" # Input label data file name
        self.label_path1 = os.path.join(self.root_dir, self.label_dir)
        self.label_path = os.listdir(self.label_path1)

        self.L0_dir = "L0_all" # Input low frequency model data file name
        self.L0_path1 = os.path.join(self.root_dir, self.L0_dir)
        self.L0_path = os.listdir(self.L0_path1)

    def __getitem__(self, item):
        seis_name = self.seis_path[item]
        seis_name_mat = seis_name.strip('.mat')
        seis_item_path = os.path.join(self.root_dir, self.seis_dir, seis_name)
        seis = scio.loadmat(seis_item_path)[seis_name_mat]
        seis = np.transpose(seis)
        seis = seis[None, ...]

        label_name = self.label_path[item]
        label_name_mat = label_name.strip('.mat')
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        label = scio.loadmat(label_item_path)[label_name_mat]
        label = np.transpose(label)
        label = label

        L0_name = self.L0_path[item]
        L0_name_mat = L0_name.strip('.mat')
        L0_item_path = os.path.join(self.root_dir, self.L0_dir, L0_name)
        L0 = scio.loadmat(L0_item_path)[L0_name_mat]
        L0 = np.transpose(L0)
        L0 = L0
        return np.array(seis, dtype=np.float32),np.array(label, dtype=np.float32),np.array(L0, dtype=np.float32)

    def __len__(self):
        return len(self.seis_path)
