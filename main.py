import numpy as np
from model import HandyFunctions
from model import Net
from model import MyDataset
import csv
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

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

csvFile = open('train.csv', 'r')
reader = csv.reader(csvFile)
data_or = []
for i in reader:
    data_or.append(i)

test_input_or = np.array(data_or)[1:, 1]
test_label_or = np.zeros(len(test_input_or))
test_label_or = np.array(test_label_or, dtype='float')
test_ecode = tmp.OneHotEncoding(test_input_or, 400)
test_ecode = torch.from_numpy(test_ecode).view(-1, 1, 14, 4)
test_dataset = MyDataset(test_ecode, test_label_or)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

net = Net()
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == "__main__":
    loss_pres = []
    for epoch in range(10):
        running_loss = 0.0
        loss_epoch = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            loss_epoch += loss.data[0]
            if i%10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss/10))
                running_loss = 0.0
        loss_pres.append(loss_epoch/500)

    plt.plot(loss_pres)
    plt.savefig("loss.png")
    plt.show()
    print('Finish Training')
    print('=========>Testing')

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        output = net(inputs)
        buf = np.array(output.data)
        for k in range(4):
            if buf[k] > 0.5:
                test_label_or[k + i*4] = 1
            else:
                test_label_or[k + i*4] = 0

    np.savetxt('pred.csv', test_label_or, delimiter=',')
    print('Finish')



