import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

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


# loading based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

train_and_valid_data = FashionMNIST("./data", train=True, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))
test_data = FashionMNIST("./data", train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))

batch_size = 256
split = 10000

num_train = len(train_and_valid_data)
indices = list(range(num_train))

np.random.seed(50676)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader( train_and_valid_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader( train_and_valid_data, batch_size=batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
	def __init__(self, dimensions):
		super(MLP, self).__init__()
		self.layers =  nn.ModuleList()
		self.length = len(dimensions)-1
		for i in xrange(self.length):
			self.layers.append( nn.Linear(dimensions[i], dimensions[i+1]) )
		self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
		
	def forward(self, image):
		batch_size = image.size()[0]
		x = image.view(batch_size, -1)
		for i in xrange(self.length-1):
			x = F.sigmoid(self.layers[i](x))
		x = F.log_softmax(self.layers[self.length-1](x), dim=0)
		return x



def train(model):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data.cuda(),), Variable(target.cuda(),)
		model.optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		model.optimizer.step()


def test(model, loader, name):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in loader:
		data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(),)
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum u
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max l
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	test_loss /= 10000
	print  name , "set : Average loss:", test_loss, " Accuracy:", correct, "/", 10000, "=", 100. * correct / 10000, "%"
	return test_loss, 100. * correct / 10000


learning_rate = 0.05
architectures = [ [28*28, 10],
                [28*28, 64, 10], [28*28, 512, 10], [28*28, 1024, 10],
  							[28*28, 64, 64, 10], [28*28, 512, 64, 10], [28*28, 512, 512, 10], [28*28, 1024, 512, 10],
  							[28*28, 64, 512, 10], [28*28, 512, 1024, 10], ]

# redirect print to log file : https://stackoverflow.com/questions/2513479/redirect-prints-to-log-file
old_stdout = sys.stdout
log_file = open("experiences.log","w")
sys.stdout = log_file



for arc in architectures:
	epochs = 350
	print(arc)
	
	RdN = MLP(arc)
	RdN.cuda()
	
	losses =[]
	
	for epoch in range(1, epochs + 1):
		train(RdN)
		loss, accuracy = test(RdN, valid_loader, 'valid')
		losses.append(loss)
	loss, accuracy = test(RdN, test_loader, 'test')
	
	plt.title("Architecture : " + str(arc) + "  & Learning rate : " + str(learning_rate) + " (accuracy : " + str(accuracy)[:4] + ") ")
	plt.ylabel("Average negative log likelihood")
	plt.xlabel("Epoch")
	plt.plot(losses, label="validation")
	plt.legend()
	plt.savefig(str(arc) + ".png") #plt.show()
	plt.close()

log_file.close()







