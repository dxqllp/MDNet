import os
import random
from math import floor

import torch
from PIL import Image
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import xml.etree.ElementTree as ET
transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#构建数据集
class MyDataset(Dataset):

    def __init__(self, path,model):
        self.path = path
        self.model=model
        self.imgs = []
        self.ssw_txt = open(path+'/'+model+'/'+model+'.txt', encoding='utf-8').readlines()
        for line in self.ssw_txt:
            line = line.rstrip()
            name = line.split()[0]
            label = Variable(Tensor(3).fill_(0.0), requires_grad=False)
            label_path = os.path.join(self.path,self.model,'labels', name.split('.')[0] + '.xml')
            tree = ET.parse(label_path)  # 读取xml文档
            for obj in tree.findall("object"):
                class_num = int(obj.find("name").text) - 1
            label[class_num] = 1
            proposal = line.split()[1::]
            ssw_block = torch.Tensor(floor((len(proposal)) / 4), 4)
            for th in range(ssw_block.shape[0]):
                ssw_block[th, 0] = int(proposal[th * 4])
                ssw_block[th, 1] = int(proposal[th * 4 + 1])
                ssw_block[th, 2] = int(proposal[th * 4 + 2])
                ssw_block[th, 3] = int(proposal[th * 4 + 3])
            self.imgs.append([name, ssw_block, label])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        name = self.imgs[index][0]
        label = self.imgs[index][2]
        image=Image.open(os.path.join(self.path,self.model,'images',self.imgs[index][0]))
        ssw = self.imgs[index][1]
        return transforms(image),ssw,label,name
