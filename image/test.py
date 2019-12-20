import torch 
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
import myDataset
from torch.utils.data import Dataset, DataLoader
from myNet import CNNnet
import sys

if __name__=="__main__":
    # 数据集的预处理
    data_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5],[0.5])
        ]
    )
    batch_size = 64
    if len(sys.argv) > 2:
        print("error argv!")
        exit()
    elif len(sys.argv) == 1:
        test_data = myDataset.fingerprintDataset(train=False, transform=data_tf)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    else:
        if sys.argv[1]=="random":
            testclass = 1
        elif sys.argv[1]=="fgsm":
            testclass = 2
        elif sys.argv[1]=="fool":
            testclass = 3
        else:
            print("error argv!")
            exit()
        test_data = myDataset.fingerprintDataset(train=False, transform=data_tf, test=testclass)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 测试网络
    model = CNNnet()
    model.load_state_dict(torch.load(r'./model'))
    model.to(device)
    print("test start")
    accuracy_sum = []
    for i,(test_x,test_y) in enumerate(test_loader):
        test_x = Variable(test_x)
        test_x = test_x.to(device)
        test_y = Variable(test_y)
        test_y = test_y.to(device)
        out = model(test_x)
        # print('test_out:\t',torch.max(out,1)[1])
        # print('test_y:\t',test_y)
        accuracy = torch.max(out,1)[1].cpu().numpy() == test_y.cpu().numpy()
        accuracy_sum.append(accuracy.mean())
        #print('accuracy:\t',accuracy.mean())
        #print(torch.max(out,1)[1].cpu().numpy())
        #print(test_y.cpu().numpy())
    print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
