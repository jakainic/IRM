import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torch.utils.data import random_split, Dataset

#for now: make training environments of relatively equal size
#would be interesting to explore in future what happens if that is not the case
#probs in order train**, test 
def data_to_envs(root, env_probs):
	data = datasets.MNIST(root, train=True, download=True)
	
	num_images = len(data)
	num_envs = len(env_probs)
	test_size = num_images//5
	train_env_size = (num_images-test_size)//(num_envs-1)

	sizes = [train_env_size for i in range(num_envs-1)]
	#check to make sure this won't throw things off too much
	sizes.append(num_images-sum(sizes))
	
	subsets = random_split(data, sizes)
	num_train_envs = len(subsets)-1

	train_envs = {'TrainEnv'+str(i): build_environment(subsets[i], env_probs[i]) for i in range(num_train_envs)}
	test_env = build_environment(subsets[-1], env_probs[-1])
	
	return train_envs, test_env

def build_environment(subset, flip_prob):
	images = subset.dataset.data[subset.indices]
	digits = subset.dataset.targets[subset.indices]

	uniform = torch.rand(len(images))
	labels = (digits<5).float()
	labels = torch.where(uniform<0.25, 1-labels, labels)

	uniform = torch.rand(len(images))
	colors = torch.where(uniform<flip_prob, 1-labels, labels)

	return ColoredMNIST(color_images(images, colors), labels)

def color_images(images, is_Red):
	#subsampling for efficiency
	images = images[:,::2,::2]
	num_images = images.shape[0]
	h,w = images[0].shape

	#add 3 color channels
	#blue is always 0
	images = torch.stack((images,images,torch.zeros(num_images,h,w,dtype=torch.int)),dim=1)

	#remove unwanted color
	for i, image in enumerate(images):
		if is_Red[i]:
			images[i][1:,:,:]*=0
		else:
			images[i][0][:,:]*=0

	return images.float()

def display_images(environment, environment_name):
	fig, axs = plt.subplots(3,4)
	fig.suptitle(environment_name)
	
	for i,ax in enumerate(axs.flatten()):
		#for pyplot, must permute dimensions so channels are last
		image = environment['images'][i].permute(1,2,0)
		ax.set_title("Label: " + str(environment['labels'][i]))
		ax.imshow(image)

	plt.show()

class ColoredMNIST(Dataset):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
	
	def __len__(self):
		return len(self.labels)
	def __getitem__(self, index):
		return self.images[index], self.labels[index]