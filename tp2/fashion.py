import numpy as np
import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from torchvision.datasets.mnist import MNIST

# to install pytorch in temp with no admin right
# pip install --install-option="--prefix=/tmp/" --user http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
# pip install --install-option="--prefix=/tmp/" torchvision


class FashionMNIST(MNIST):
	"""`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
	Args:
		root (string): Root directory of dataset where ``processed/training.pt``
			and  ``processed/test.pt`` exist.
		train (bool, optional): If True, creates dataset from ``training.pt``,
			otherwise from ``test.pt``.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
	"""
	urls = [
			'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
			'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
			'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
			'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
			]



train_data = FashionMNIST("./data", train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))
valid_data = FashionMNIST("./data", train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))



class layer(object):
	def __init__(self, nIn, nOut): # Notre methode constructeur		
		self.w = np.random.random_integers(-3,3, (nIn,nOut))
		self.b = np.random.random_integers(-3,3, (1,nOut))
	def fprop(self, x):
		z = np.dot(x, self.w) + self.b
		return z
	def update(self, lr, x, dEdy):
		dEdw = np.dot(x.T, dEdy)
		dEdb = np.mean(dEdy, axis = 0)		
		self.w = self.w - dEdw * lr
		self.b = self.b - dEdb * lr
		dEdx = np.dot(dEdy,self.w.T)
		return dEdx


class MLP(object):
	def __init__(self, list):
		self.L = layer(list[0],list[1])
		self.M = layer(list[1],list[2])
	def fprop(self, x):
		print np.shape(x.shape)
		self.cache = self.L.fprop(x)
		return self.M.fprop(self.cache)
	def update(self, lr, x, t):
		dEdy = 2*(self.fprop(x) - np.eye(10)[t])
		self.L.update(lr, x, self.M.update(lr, self.cache, dEdy))
		return
	def error(self, x, t):
		z = np.argmax(self.fprop(x), axis = 1)
		errors = np.sum(z!=t)
		return errors


		
RdN = MLP([ 784, 100, 10 ])
batch = 1000 # nbr dexemples appris a la fois
epochs = 10

# mesuring time
start = time.time()
for i in range(epochs):
	
	# training
	for j in range(len(train_data)/batch):
		RdN.update(.00001, np.array(train_data.train_data[j*batch:(j+1)*batch].numpy().reshape(784,1000).T), np.array(train_data.train_labels[j*batch:(j+1)*batch]))
	
	# mesuring error rate
	error_count = 0
	for j in range(len(valid_data)/batch):
		error_count += RdN.error(valid_data.test_data[j*batch:(j+1)*batch], valid_data.test_labels[j*batch:(j+1)*batch])
	error_rate = error_count / len(train_data)
	
	# report
	print "validation error rate : " + str(error_rate) + " epoch " + str(i) 
print " execution time : " + str(time.time() - start)
















