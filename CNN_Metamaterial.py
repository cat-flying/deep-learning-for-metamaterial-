import torch as t
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import datetime

'''a = t.cuda.is_available()
print(a)
ngpu= 1
device = t.device("cuda:0" if (t.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(t.cuda.get_device_name(0))'''



start = datetime.datetime.now()#计算程序开始时间
LR = 0.0008
EPOCH = 1000
BATCH_SIZE = 64
d_set = np.zeros(shape=(2400,1000))#训练集容量
d_test = np.zeros(shape=(100,1000))
x1 = np.arange(0, 1000)
for i in range(1,2401):
    filename = 'D:/AI/traing_2/' + str(i) + '.txt'
    data = np.array(np.loadtxt(filename))#读取训练数据文件
    d_set[i-1] = data#将训练文件储存在d_set中
x_train = t.from_numpy(d_set)#将训练文件变成张量
x_train = x_train.float()
x_train = x_train.reshape(2400,1,1000)
#读取验证文件
for i in range(100):
    file= 'D:/AI/test_2/' + str(i+2401) + '.txt'
    dat = np.array(np.loadtxt(file))
    d_test[i] = dat

x_test = t.from_numpy(d_test)
x_test = x_test.float()
x_test = x_test.reshape(100,1,1000)


data = np.loadtxt('cnn_data.txt')
data_y = data[0:2400]
test_y = data[2400:2500]
y_train = t.from_numpy(data_y)
y_test = t.from_numpy(test_y)
y_test = y_test.float()
y_test = y_test.reshape(100,1,32)
y_train = y_train.float()
y_train = y_train.reshape(2400,1,32)#改变data_y的维度，使其变成三维的
#print(x_train.size())#size函数返回张量的尺寸
#print(x_train)
x_train = x_train.cuda()#使用GPU进行加速
y_train = y_train.cuda()#使用GPU进行加速

torch_dataset = Data.TensorDataset(x_train,y_train)#将张量的训练数据与预测数据输入到模型的数据集中
test_data = Data.TensorDataset(x_test,y_test)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False
)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,#批量训练大小
    shuffle=False#true打乱数据进行训练
)
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #第一个卷积层以及其参数

        self.conv1 = nn.Sequential(
            #nn.BatchNorm1d(1, momentum=0.2),
            nn.Conv1d(
                in_channels=1, #前面输入数据的信道数是多少
                out_channels=6, #输出的信道数，即卷积核的个数
                kernel_size=3,
                stride = 1,
                padding=1, 
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2),
        )#output的数据维度大小为（(n+2*p-f)/s +1）
        #第二个卷积层以及其参数#前面一层输出的个数为500

        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(6,momentum=0.5),
            nn.Conv1d(
                in_channels= 6 ,
                out_channels = 3,
                kernel_size= 3 ,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(3,momentum=0.5),
            nn.Conv1d(
                in_channels=3,
                out_channels=1,
                kernel_size = 3,
                stride = 1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
        )#最后输出的个数为124个

        self.out = nn.Sequential(
            nn.BatchNorm1d(1,momentum=0.5),
            nn.Linear(125,100),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1, momentum=0.5),
            #nn.Linear(100,96),
            #nn.Dropout(0.5),
            #nn.LeakyReLU(),
            nn.BatchNorm1d(1, momentum=0.5),
            nn.Linear(100, 96),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1, momentum=0.5),
            nn.Linear(96, 64),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1, momentum=0.5),
            nn.Linear(64,32),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        output = self.out(x3)
        return output

cnn = CNN()
cnn.cuda()#把模型移动到GPU中
optimizer = t.optim.Adam(cnn.parameters(),lr = LR,weight_decay=0.0015)
loss_func = nn.MSELoss()

#训练过程
Loss_list = []
loss1 = []
xx = []
xx1 = []
num = 0
test_num = 0
for epoch in range(EPOCH):
    for step, (x,y) in enumerate(loader):


        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()
        #b_x = t.tensor(b_x, dtype=t.float32)
        #b_y = t.tensor(b_y, dtype=t.float32)
        pred = cnn(b_x)
        loss = loss_func(pred,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    Loss_list.append((loss.data.cpu().numpy()).tolist())
    num = num+1
    xx.append(num)

    if epoch % 10 == 0:#计算训练集的损失
        for step, (x1, y1) in enumerate(test_loader):
            test_x = Variable(x1).cuda()
            test_y = Variable(y1).cuda()
            pred = cnn(test_x)
            loss2 = loss_func(pred,test_y)
        loss1.append((loss2.data.cpu().numpy()).tolist())
        xx1.append(epoch)
end = datetime.datetime.now()


print('训练模型所花时间为：',(end - start).seconds)

t.save(cnn.state_dict(),'cnn.pth')
plt.plot(xx,Loss_list)
plt.plot(xx1,loss1,'blue')
plt.show()
'''for i,l_his in enumerate(Loss_list):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()'''






