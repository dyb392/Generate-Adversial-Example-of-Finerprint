import torch 
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
import myDataset
from torch.utils.data import Dataset, DataLoader

# 数据集的预处理
data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]
)

batch_size = 64
train_data = myDataset.fingerprintDataset(train=True, transform=data_tf)
test_data = myDataset.fingerprintDataset(train=False, transform=data_tf)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=2,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,2,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,2,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(38400,1000)
        self.mlp2 = torch.nn.Linear(1000,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x
    
if __name__=="__main__":    
    model = CNNnet()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(model)

    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=0.01)

    model = model.to(device)
    loss_func = loss_func.to(device)

    epochs = 20
    loss_count = []
    for epoch in range(epochs):
        for i,(x,y) in enumerate(train_loader):
            batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
            batch_y = Variable(y) # torch.Size([128])
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 获取最后输出
            out = model(batch_x) # torch.Size([128,10])
            # 获取损失
            batch_y = batch_y.long()
            loss = loss_func(out,batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            if i%20 == 0:
                loss_count.append(loss)
                print('{}:\t'.format(i), loss.item())
                #torch.save(model,r'./model')
            if i % 50 == 0:
                for a,b in test_loader:
                    test_x = Variable(a)
                    test_y = Variable(b)
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    out = model(test_x)
                    # print('test_out:\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    accuracy = torch.max(out,1)[1].cpu().numpy() == test_y.cpu().numpy()
                    print('accuracy:\t',accuracy.mean())
                    break
        print("train: ",epoch, "/", epochs)

    torch.save(model.state_dict(), r'./model')

    # 测试网络
    test_model = CNNnet()
    test_model.load_state_dict(torch.load(r'./model'))
    test_model = test_model.to(device)

    accuracy_sum = []
    for i,(test_x,test_y) in enumerate(test_loader):
        test_x = Variable(test_x)
        test_x = test_x.to(device)
        test_y = Variable(test_y)
        test_y = test_y.to(device)
        out = test_model(test_x)
        # print('test_out:\t',torch.max(out,1)[1])
        # print('test_y:\t',test_y)
        accuracy = torch.max(out,1)[1].cpu().numpy() == test_y.cpu().numpy()
        accuracy_sum.append(accuracy.mean())
        #print('accuracy:\t',accuracy.mean())
        #print(torch.max(out,1)[1].cpu().numpy())
        #print(test_y.cpu().numpy())
    print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))

