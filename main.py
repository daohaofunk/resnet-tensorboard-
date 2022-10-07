import os
import csv
import random
import shutil

import torch
import torch.nn as nn
import torch.backends.cuda as cuda
import torch.optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#----------------将数据制作标签
def make_label():
    label={}
    # 数据地址，这里需要更改为自己的地址，比如我的地址是F:/pythondemo/data/Rice Leaf Disease Images，全代码一共有1/5需要更改
    # os.getcwd()会返回程序当前地址，所以要把主程序和data放在同级文件夹
    data_dir = os.path.join(os.getcwd(),'data','Rice Leaf Disease Images')
    for file in os.listdir(data_dir):
        temp_dir = os.path.join(data_dir,file)
        for img in os.listdir(temp_dir):
            name = img.split('.')[0]
            label[name] = file
    header = ['id','class']
    csv_dir = os.path.join(data_dir,'label.csv')
    with open(csv_dir,'w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=header)
        writer.writeheader()
        for i in range(len(label)):
            temp_dic = {}
            temp_dic['id'] = list(label.keys())[i]
            temp_dic['class'] = list(label.values())[i]
            writer.writerow(temp_dic)
    return label
if  os.access(os.path.join(os.getcwd(),'data','Rice Leaf Disease Images','label.csv'), os.F_OK):#如果已经完成创建label则跳过，地址更改2/5
    pass
else:
    make_label()
def copyfile(filename, target_dir):
    """Copy a file into a target directory.

    Defined in :numref:`sec_kaggle_cifar10`"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
#----------------按照格式创建train和test集
def make_set():
    data_dir = os.path.join(os.getcwd(), 'data', 'Rice Leaf Disease Images')#地址更改3/5
    for dic in os.listdir(data_dir):
        if dic != 'label.csv':
            temp_dir = os.path.join(data_dir, dic)
            for img in os.listdir(temp_dir):
                r = random.random()
                # 这里依照概率随机分选训练集和测试集，设置为train：test =9:1
                if r <=0.9:
                    fname = os.path.join(data_dir, dic,img)
                    copyfile(fname, os.path.join(data_dir, 'train'))
                else:
                    fname = os.path.join(data_dir, dic, img)
                    copyfile(fname, os.path.join(data_dir, 'test'))
if  os.access(os.path.join(os.getcwd(),'data','Rice Leaf Disease Images','train'), os.F_OK):#如果已经完成set创建则跳过，地址更改4/5
    pass
else:
    make_set()
#-----------------数据分组
batchsize = 32
valid_ratio = 0.2
data_dir = os.path.join(os.getcwd(),'data','Rice Leaf Disease Images')#地址更改5/5
def reorg_data(data_dir,valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir,'label.csv'))
    d2l.reorg_train_valid(data_dir,labels,valid_ratio)
    d2l.reorg_test(data_dir)
reorg_data(data_dir,valid_ratio)
#-----------------数据增强
transform_train = transforms.Compose([
    #随机裁剪
    transforms.RandomResizedCrop(224),
    #水平翻转
    transforms.RandomHorizontalFlip(),
    #随机亮度饱和度
    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    #随机噪声
    transforms.ToTensor(),
    #标准化图层
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

transforms_test = transforms.Compose([
    #增大至256再中心裁剪
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
#--------------------数据集加载
train_set,train_valid_set = [ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transform_train) for folder in ['train','train_valid']]
valid_set,test_set = [ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transforms_test) for folder in ['valid','test']]
#--------------------数据集迭代实例
train_iter,train_valid_iter = [DataLoader(dataset,batchsize,shuffle=True,drop_last=True) for dataset in (train_set,train_valid_set)]
valid_iter = DataLoader(valid_set,batchsize,shuffle=True,drop_last=True)
test_iter = DataLoader(test_set,batchsize,shuffle=True,drop_last=True)
#--------------------模型预训练及全连接层改写
def get_net(device):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 修改最后的全连接层,这里中的256可以改成任意数，最后的5是你的类别数
    finetune_net.output_new = nn.Sequential(nn.Linear(1000,256),
                                            nn.ReLU(),
                                           nn.Linear(256,5))
    # xavier初始化最后一层参数，但是我自己试了效果一般
    # nn.init.xavier_normal_(finetune_net.output_new[0].weight)
    # nn.init.xavier_normal_(finetune_net.output_new[3].weight)
    # 选择cpu还是gpu
    finetune_net = finetune_net.to(device)
    # 冻结resnet前面参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
#--------------------训练gpu设置
# 这两个函数允许我们在请求GPU不存在的情况下运行代码
    """如果存在GPU就返回GPU(i)，否则返回cpu()"""
def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(i)
    return torch.device('cpu')
device = try_gpu()
#--------------------最后两层的loss
loss = nn.CrossEntropyLoss(reduction='none')#对于分类模型通常选用交叉熵损失
def evaluate_loss(data_iter,net,device):
    l_sum,n = 0.0,0
    for features,labels in data_iter:
        features,labels = features.to(device),labels.to(device)
        outputs = net(features)
        l = loss(outputs,labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum/n).to(device)
#--------------------可视化训练
def visiable_train(iter,train_loss=None,test_loss=None):
    writer = SummaryWriter('log/log1')
    if train_loss:
        writer.add_scalar(tag='train loss',scalar_value=train_loss,global_step=iter)
    if test_loss:
        writer.add_scalar(tag='test_loss',scalar_value=test_loss,global_step=iter)
    writer.close()
#-------------------训练函数
def train(net,train_iter,test_iter,num_epochs,lr,wd,device,lr_period,lr_decay):
    # 只训练最后的全连接层
    net = net.cuda(device)
    trainer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad),lr=lr,momentum=0.9,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer,lr_period,lr_decay)
    num_batches,timer = len(train_iter),d2l.Timer()
    legend = ['train loss','train acc','test acc']
    animator  = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0,1],legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(device), labels.to(device)
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            nn.utils.clip_grad_value_(net.parameters(), clip_value=1.1)#梯度裁剪
            trainer.step()
            train_acc_sum = d2l.accuracy(output, labels)
            metric.add(l,train_acc_sum, labels.shape[0],labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3], None))
            visiable_train(iter=i,train_loss=(metric[0] / metric[2]))#可视化loss
        scheduler.step()
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {metric[0] / metric[2]:.3f}'
          f'train acc {metric[1] / metric[3]:.3f}'
          f'test acc {test_acc:.3f}')
    print(f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                     f' examples/sec on {str(device)}')
#----------------------训练模型
if __name__ =='__main__':
    num_epochs, lr, wd = 10, 1e-4, 1e-4
    lr_period, lr_decay, net = 2, 0.9, get_net(device)
    train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period,lr_decay)
    plt.figure()
    plt.show()