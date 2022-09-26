from environments import *
from model import *
from train_test import *

full_train = True

dim1 = 256
dim2 = 256
reg_wt = 0.001
lr = 0.001
penalty_wt = 0
num_epochs = 501 if full_train else 101
anneal_epoch = 0
batch_size = 64

train_envs, test_env = data_to_envs('~/datasets/mnist',[0.1,0.2,0.9])
#display_images(train_envs['TrainEnv0'], "First Train Environment")
#display_images(test_env, "Test Environment")

config = {"reg_wt":reg_wt, "lr": lr, "penalty_wt": penalty_wt, "dim1": dim1,"dim2":dim2, "num_epochs":num_epochs, "batch_size":batch_size, "anneal_epoch":anneal_epoch}
if full_train:
	train_full(config, train_envs, 'here_be_model.pt')
else:
	train_minibatches(config, train_envs, 'here_be_model.pt')
	
#test(config, test_env, 'here_be_model.pt')