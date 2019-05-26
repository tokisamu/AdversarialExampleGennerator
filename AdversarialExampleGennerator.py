from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import TestModel
import Train
adversarialRate = [0,0.1,0.2,0.3,0.5,1,10]
model = TestModel.getInitModel()
#正则化数据
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)

# 训练
accuracy = []
ans = []
for i in adversarialRate:
    acc, ex = Train.train(model, "cpu", test_loader, i)
    accuracy.append(acc)
    ans.append(ex)