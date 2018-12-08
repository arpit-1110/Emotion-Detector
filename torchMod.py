import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.image as img
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt 
import numpy as np 
import sys
import os 

class FER2013(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		curr_path = os.getcwd()
		data = pd.read_csv(curr_path+'/fer2013/fer2013.csv')
		data = np.array(data)
		output = data[:, 0]
		y = np.zeros((len(output), 7))
		for i in range(len(output)):
		    y[i][output[i]] = 1
		X = data[:, 1]
		X = [np.fromstring(x, dtype='int', sep=' ') for x in X]
		X = np.array([np.fromstring(x, dtype='int', sep=' ').reshape(2304)
		              for x in data[:, 1]])
		final = np.zeros((X.shape[0], X.shape[1] + y.shape[1]))
		final[:,:X.shape[1]] = X
		final[:,X.shape[1]:X.shape[1]+y.shape[1]] = y 
		self.file_list = final 
		self.transform = transform

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		data = self.file_list[idx]
		X = data[:-1]
		y = data[-1]
		train_set = [torch.from_numpy(X).float(), torch.from_numpy(y).float()]
		return train_set

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(, kernel_size=3, stride=1, padding=1)
		