# resnet-tensorboard-

## 前言

本文针对刚刚了解深度学习并希望快速上手的同学，将对代码各个模块的内容做简单的讲解。代码有点长，只做部分讲解

## 深度学习训练框架

+ 给自己的数据集制作标签
+ 对数据集进行训练集测试集分组
+ 如果数据是小样本，需要进行数据增强（扩充样本集）
+ 数据集加载
+ 数据集迭代器
+ 对resnet网络输出层进行全连接更改
+ 训练网络
  
  ## 深度学习训练流程
  
  这里给刚入门深度学习的同学一个简单的官方深度学习实例：
  
  ```python
  def train(data):
    #将原始特征（数据），标签输入以tensor（张量）形式传入gpu中
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    # 对输入进行前向传播计算每次的back
    outputs = model(inputs)
    #根据你自己选定的损失函数计算loss（通俗讲就是和正确分类的偏差值）
    loss = criterion(outputs, labels)
    #这步是将之前的梯度归零，为下次梯度更新做准备
    optimizer.zero_grad()
    #进行反向传播，也就是训练的过程
    loss.backward()
    #进行参数的梯度的更新
    optimizer.step()
  ```
  
  # 正式代码结构部分
  
  ## 1.数据标签制作
  
  **我这里用的水稻病害的数据集，一共四种病害类别**
  你的数据结构应该像下图，按照分类将每个照片归入类的文件夹，下图展示了制作标签和数据分类之后的文件夹结构
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/4895bb48b16a487db2c4321b55e095ad.png)
  
  ## 2.对数据集进行分组
  
  其中train用来训练，迭代多次，valid用来选择超参数（比如确定模型，学习率），test数据集用于测试，只使用一次。
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/fe588787be3e4a57a763586cd15f2532.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/cf27e699d2814a019655b3c0be5e351b.png)**在完成上述操作后你的文件夹会生成三个新文件（train，test和label.csv）**

## 3.数据增强

扩充数据集，同时把你的照片格式统一为resnet的输入尺寸

## 4.数据集加载+数据迭代器

这部分用的是torchvision自带的模块，DateLoder和ImageFolder

## 5.迁移学习resnet cifar10参数并改写网络最后全连接层

由于从头训练网络需要大量的样本及迭代次数，因此我们对网络进行预训练。通过网上下载resnet对cifar10数据集训练到的提取特征的参数，可以大大减少我们的网络收敛的时间。如果需要使用gpu训练则需要下载cudnn和cuda

```python
def get_net(device):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 修改最后的全连接层，这里中的256可以改成任意数，最后的5是你的类别数
    finetune_net.output_new = nn.Sequential(nn.Linear(1000,256),
                                            nn.ReLU(),
                                           nn.Linear(256,5))
    # 选择cpu还是gpu
    finetune_net = finetune_net.to(device)
    # 冻结resnet前面参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(i)
    return torch.device('cpu')
device = try_gpu()
```

## 6.添加可视化组件，首先需要安装tensorboard（pip安装）

如果需要用gpu训练，需要下载python版本对应的cuda 和cudnn
![在这里插入图片描述](https://img-blog.csdnimg.cn/ff3a1c6cf11f461482ccfe651394415a.png#pic_center)

```python
def visiable_train(iter,train_loss=None,test_loss=None):
    writer = SummaryWriter('log/log1')
    if train_loss:
        writer.add_scalar(tag='train loss',scalar_value=train_loss,global_step=iter)
    if test_loss:
        writer.add_scalar(tag='test_loss',scalar_value=test_loss,global_step=iter)
    writer.close()
```

## 7.大功告成，训练网络

tensorboard使用官方教程

> https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
> 安装好进入你的运行环境，cmd进入主程序文件夹后输入tensorboard --logdir=log
> （log为之前设置的日志文件名）然后浏览器输入127.0.0.1:6060![在这里插入图片描述](https://img-blog.csdnimg.cn/a3c6de6f2aa44ff7bf1eb97a8e054e7b.png)

## 参考部分及帮助阅读文档

> [数据集分组](https://blog.csdn.net/qq_24884193/article/details/104071664)https://blog.csdn.net/qq_24884193/article/details/104071664
> [tensorboard使用明细](https://blog.csdn.net/weixin_41809530/article/details/111253479)https://blog.csdn.net/weixin_41809530/article/details/111253479
> [梯度裁剪](https://blog.csdn.net/CVSvsvsvsvs/article/details/91137997)https://blog.csdn.net/CVSvsvsvsvs/article/details/91137997
> [resnet网络最后全连接层调整](https://zhuanlan.zhihu.com/p/35890660)https://zhuanlan.zhihu.com/p/35890660
> [cudnn和python版本对应](https://blog.csdn.net/caiguanhong/article/details/112184290)https://blog.csdn.net/caiguanhong/article/details/112184290
