import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import argparse
import utils
from CK import CK
from torch.autograd import Variable
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()

cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

testset = CK(path = './data/test_data.h5', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('Fer2013_VGG19', 'Test_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()

net.eval()
PrivateTest_loss = 0
correct = 0
total = 0
tloss=0
total_predicted=[]
ground_truth=[]

for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
    bs, ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)

    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    outputs = net(inputs)
    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

    _, predicted = torch.max(outputs_avg.data, 1)
    
    total_predicted.extend(predicted.cpu().numpy().tolist())
    ground_truth.extend(targets.cpu().numpy().tolist())


from sklearn.metrics import f1_score
f1 = f1_score(ground_truth, total_predicted, average='macro' )
print("F1-SCORE:",f1)
from sklearn.metrics import accuracy_score
print("acc:",accuracy_score(ground_truth, total_predicted))