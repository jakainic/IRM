from environments import *
from model import *
from train_test import *

full_train = False

#for now, don't worry about hyperparameter tuning 
#use parameters from authors
reg_wt = 0.00110794568
lr = 0.0004898
penalty_wt = 91257
dim1 = 390
dim2 = 390
num_epochs = 501 if full_train else 101
anneal_epoch = 0 if full_train else 0
#will only matter if training using mini-batches
batch_size = 64

train_envs, test_env = data_to_envs('~/datasets/mnist',[0.1,0.2,0.9])
#display_images(train_envs['TrainEnv0'], "First Train Environment")
#display_images(test_env, "Test Environment")

config = {"reg_wt":reg_wt, "lr": lr, "penalty_wt": penalty_wt, "dim1": dim1,"dim2":dim2, "num_epochs":num_epochs, "batch_size":batch_size, "anneal_epoch": anneal_epoch}
if full_train:
	train_full(config, train_envs, test_env,'here_be_model.pt')
else:
	train_minibatches(config, train_envs, test_env,'here_be_model.pt')

#test(config, test_env, 'here_be_model.pt')