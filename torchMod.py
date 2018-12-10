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
import pandas as pd 

class FER2013(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		curr_path = os.getcwd()
		data = pd.read_csv(curr_path+'/fer2013/fer2013.csv')
		data = np.array(data)
		y = data[:, 0]
		# y = np.zeros((len(output), 7))
		# for i in range(len(output)):
		#     y[i][output[i]] = 1
		X = data[:, 1]
		X = [np.fromstring(x, dtype='int', sep=' ') for x in X]
		X = np.array([np.fromstring(x, dtype='int', sep=' ').reshape(2304)
		              for x in data[:, 1]])
		# final = np.zeros((X.shape[0], X.shape[1] + y.shape[1]))
		# final[:,:X.shape[1]] = X
		# final[:,X.shape[1]:X.shape[1]+y.shape[1]] = y 
		X = X.reshape(-1, 1, 48, 48)
		self.X = X
		self.y = y
		self.transform = transform

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		X = self.X[idx]
		y = self.y[idx]
		train_set = torch.from_numpy(np.asarray(X)).float(), torch.from_numpy(np.asarray(y)).long()
		return train_set

class FerNet(nn.Module) :
	def __init__(self) :
		super(FerNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, 5, stride=1)
		self.conv2 = nn.Conv2d(16, 32, 5, stride=1)
		self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(64, 128, 1, stride=1, padding=1)
		self.fc1 = nn.Linear(9*9*32, 4096)
		self.fc2 = nn.Linear(4096, 256)
		self.fc3 = nn.Linear(256, 7)

	def forward(self, x) :
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		# x = F.relu(self.conv3(x))
		# x = F.max_pool2d(x, 2, 2)
		# x = F.relu(self.conv4(x))
		# x = F.max_pool2d(x, 2, 2)
		# print(x.shape)
		x = x.view(-1, 9*9*32)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x), dim=1)
		return x


if __name__ == '__main__':

	trainset = FER2013()
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
	net = FerNet()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

	for epoch in range(5) :
		running_loss = 0 

		for i, data3 in enumerate(trainloader):
			inputs, expected = data3
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, expected)
			# print(loss)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 2000 == 1999:    
				print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0
	print("Finished Training")
	# torch.save(net.state_dict(), net.name())