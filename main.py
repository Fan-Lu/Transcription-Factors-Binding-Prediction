import numpy as np
from model import HandyFunctions
from model import Net
from model import MyDataset
import csv
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

tmp = HandyFunctions()

csvFile = open('train.csv', 'r')
reader = csv.reader(csvFile)
data_or = []
for i in reader:
    data_or.append(i)

input_or = np.array(data_or)[1:, 1]
label_or = np.array(data_or)[1:, 2]
label_or = np.array(label_or, dtype='float')
ecode = tmp.OneHotEncoding(input_or, 2000)
ecode = torch.from_numpy(ecode).view(-1, 1, 14, 4)
dataset = MyDataset(ecode, label_or)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

net = Net()
#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == "__main__":
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i%10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss/10))
                running_loss = 0.0

    print('Finish Training')
