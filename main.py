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
                                          shuffle=False, num_workers=2)

csvFile = open('test.csv', 'r')
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
                                          shuffle=False, num_workers=2)

net = Net()
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train():
    loss_pres = []
    for epoch in range(100):
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
        if epoch%10 == 9:
            torch.save(net.state_dict(), 'models/model_epoch_%d.pth' % (epoch+1))
    plt.plot(loss_pres)
    plt.savefig("loss.png")
    plt.show()
    print('Finish Training')
    print('=========>Testing')


def train_acc():
    cor = 0.0
    train_label_or = np.zeros(2000)
    net.load_state_dict(torch.load('models/model_epoch_100.pth'))
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        output = net(inputs)
        buf = np.array(output.data)
        for k in range(4):
            #train_label_or[k + i * 4] = buf[k]
            if buf[k] > 0.08:
                train_label_or[k + i * 4] = 1
            else:
                train_label_or[k + i * 4] = 0
    for j in range(1000):
        if train_label_or[j] == 1:
            cor += 1
    for j in range(1000):
        if train_label_or[j+1000] == 0:
            cor += 1
    rate = cor/2000
    np.savetxt('train_pred.csv', train_label_or, delimiter=',')
    print('Finish Train Eval')
    return rate


def test_acc():
    net.load_state_dict(torch.load('models/model_epoch_100.pth'))
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        output = net(inputs)
        buf = np.array(output.data)
        for k in range(4):
            if buf[k] > 0.08:
                test_label_or[k + i*4] = 1
            else:
                test_label_or[k + i*4] = 0

    np.savetxt('test_pred.csv', test_label_or, delimiter=',')
    print('Finish Test Prediction')


if __name__ == "__main__":
    #train()
    acc = train_acc()
    print(acc)
    test_acc()



