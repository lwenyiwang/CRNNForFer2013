#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

from skimage import io
from skimage.transform import resize
from models import *
import h5py
from tqdm import tqdm

data = h5py.File('./data/dev_data.h5', 'r', driver='core')
number = len(data['data_label'])
raw_img=[]
label=[]
for i in tqdm(range(number)):
    raw_img.append(data['data_pixel'][i])
    label.append(data['data_label'][i])

# raw_img = io.imread('images/9.jpg')
img = np.array(raw_img)
img = img.astype('float32')
img/=255
img=img.reshape(-1,1,48,48)

img = np.concatenate((img, img, img), axis=1)
print(img.shape)
batchs=img.shape[0]/100


net = VGG('VGG19')
# checkpoint = torch.load(os.path.join('FER2013_VGG19', 'modelnew428.t7'))
checkpoint = torch.load(os.path.join('CK+_VGG19/1', 'New_CRNN.t7'))

net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()
result=[]
for i in range(batchs):
    print("processing batch :",i," .............")
    if i==batchs-1:
        img2=img[i*100:img.shape[0]]
    else:
        img2=img[i*100:(i+1)*100]

    inputs=torch.from_numpy(img2).float()
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, -1)
    result+=list(predicted.cpu().numpy())
print(len(result))

from sklearn.metrics import f1_score
f1 = f1_score(label, result, average='macro' )
print("F1-SCORE:",f1)
from sklearn.metrics import accuracy_score
print("acc:",accuracy_score(label, result))
