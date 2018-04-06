import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data

class HandyFunctions:
    def OneHotEncoding(self, data, no_samples):
        A = np.array([[1.0, 0.0, 0.0, 0.0]])
        C = np.array([[0.0, 1.0, 0.0, 0.0]])
        G = np.array([[0.0, 0.0, 1.0, 0.0]])
        T = np.array([[0.0, 0.0, 0.0, 1.0]])

        len_kernel = 0
        All_S_Test = np.zeros((no_samples, 14 * 4), dtype=np.float32)

        for k in range(0, no_samples):
            in_seq = data[k]

            for j in range(0, 14):
                if in_seq[j] == 'A':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = A
                if in_seq[j] == 'C':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = C
                if in_seq[j] == 'G':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = G
                if in_seq[j] == 'T':
                    All_S_Test[k, (len_kernel + j) * 4:(len_kernel + j + 1) * 4] = T

        return All_S_Test

    def PerformanceEval(self, y_out, test_label, no_samples):
        eval_metric = np.zeros((no_samples, 4), dtype=np.float)

        return eval_metric


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #  1 input image channel, 6 output channels, 5x5 square convolution
        #  kernel
        self.conv1 = nn.Conv2d(1, 6, 1)
        self.conv2 = nn.Conv2d(6, 16, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*14*4, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 1)
        # If the size if a square, only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 1)
        x = x.view(-1, 16*14*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)