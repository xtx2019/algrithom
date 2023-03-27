import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image

features_dir = './features'  # 存放特征的文件夹路径
data_list = []  # 定义一个空的存放数据的列表
path = 'Test_image'  # 这里是存放图片的路径，所有的图片均存放RSICD_images中
for filename in os.listdir(path):  # 依次遍历读取文件夹的图片并处理
    transform1 = transforms.Compose([  # 串联多个图片变换的操作
        transforms.Resize(256),  # 缩放
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor()]  # 转换成Tensor
    )
    print(filename)
    img = Image.open(path + '/' + filename)  # 打开图片
    img1 = transform1(img)  # 对图片进行transform1的各种操作
    resnet50_feature_extractor = models.resnet50(pretrained=True)  # 导入ResNet50的预训练模型
    resnet50_feature_extractor.fc = nn.Linear(2048, 2048)  # 重新定义最后一层
    torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)  # 将二维tensor初始化为单位矩阵

    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)

    y = resnet50_feature_extractor(x)
    y = y.data.numpy()  # 将提出出来的特征向量转化成numpy形式便于后面存储
    data_list.append(y)  # 依次将每一张图片提出出来的向量放在列表中
    data_npy = np.array(data_list)
np.save('list.npy', data_list)  # 存储为.npy文件
