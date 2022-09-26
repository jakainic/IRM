import torch
from torch.utils.data import DataLoader, Dataset
from environments import ColoredMNIST
from model import *

def train_full(config, train_data, test_data, filepath):
	mlp = MLP(config["dim1"], config["dim2"])
	optimizer = torch.optim.Adam(mlp.parameters(), lr = config["lr"], weight_decay = config['reg_wt'])
	train_dataloaders = {env_name: DataLoader(train_data[env_name], batch_size = len(train_data[env_name])) for env_name in train_data.keys()}
	
	risks = {env_name: -1 for env_name in train_data.keys()}
	accuracies = {env_name: -1 for env_name in train_data.keys()}
	penalties = {env_name: -1 for env_name in train_data.keys()}

	num_envs = len(train_data.keys())

	test_dataloader = DataLoader(test_data,batch_size = len(test_data))
	test_images, test_labels = next(iter(test_dataloader))

	for epoch in range(config["num_epochs"]):
		for env_name in train_data.keys():
			images, labels = next(iter(train_dataloaders[env_name]))

			outputs = mlp(images).flatten()
			risks[env_name] = risk(outputs, labels)
			accuracies[env_name] = accuracy(outputs, labels)
			penalties[env_name] = penalty(outputs,labels)

		avg_risk = sum(risks.values())/num_envs
		avg_accuracy = sum(accuracies.values())/num_envs
		avg_penalty = sum(penalties.values())/num_envs

		penalty_wt = config["penalty_wt"] if epoch>=config["anneal_epoch"] else 1.0
		loss = avg_risk + penalty_wt*avg_penalty
		loss /= penalty_wt if penalty_wt>1.0 else 1.0

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch % 50 == 0:
			print("Epoch: " + str(epoch))
			print("Avg risk: " + str(avg_risk))
			print("Avg penalty: " + str(avg_penalty))
			print("Avg accuracy: "  + str(avg_accuracy))

			with torch.no_grad():
				test_outputs = mlp(test_images).flatten()
				test_acc = accuracy(test_outputs, test_labels)
				print("Test accuracy: " + str(test_acc))

	#save model for inference
	torch.save(mlp.state_dict(), filepath)

	return avg_risk, avg_accuracy

#FIX: this function is far too long, and lots of repeated code from above
def train_minibatches(config, train_data, test_data, filepath):
	mlp = MLP(config["dim1"], config["dim2"])
	optimizer = torch.optim.Adam(mlp.parameters(), lr = config["lr"], weight_decay = config['reg_wt'])
	
	risks = {env_name: -1 for env_name in train_data.keys()}
	accuracies = {env_name: -1 for env_name in train_data.keys()}
	penalties = {env_name: -1 for env_name in train_data.keys()}

	train_dataloaders = {env_name: DataLoader(train_data[env_name], batch_size = config["batch_size"]) for env_name in train_data.keys()}

	#for now: assuming training environments are the same size (not ideal for flexibility)
	first_env = list(train_data.keys())[0]
	num_els_per_env = len(train_data[first_env])
	num_steps = num_els_per_env//config["batch_size"]
	
	num_envs = len(train_data.keys())

	test_dataloader = DataLoader(test_data,batch_size = len(test_data))
	test_images, test_labels = next(iter(test_dataloader))

	for epoch in range(config["num_epochs"]):
		epoch_risk = 0
		epoch_penalty = 0
		epoch_accuracy = 0

		for step in range(num_steps):
			for env_name in train_data.keys():
				images, labels = next(iter(train_dataloaders[env_name]))
				outputs = mlp(images).flatten()

				risks[env_name] = risk(outputs,labels)
				accuracies[env_name] = accuracy(outputs, labels)
				penalties[env_name] = penalty(outputs, labels)


			avg_risk = sum(risks.values())/num_envs
			avg_accuracy = sum(accuracies.values())/num_envs
			avg_penalty = sum(penalties.values())/num_envs

			#rescale for running avg
			epoch_risk+= avg_risk*config["batch_size"]/num_els_per_env
			epoch_penalty+=avg_penalty*config["batch_size"]/num_els_per_env
			epoch_accuracy+=avg_accuracy*config["batch_size"]/num_els_per_env

			penalty_wt = config["penalty_wt"] if epoch<config["anneal_epoch"] else 1.0
			loss = avg_risk + penalty_wt*avg_penalty
			loss /= penalty_wt if penalty_wt>1.0 else 1.0

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch % 10 == 0:
			print("Epoch: " + str(epoch))
			print("Avg risk: " + str(epoch_risk))
			print("Avg penalty: "+str(epoch_penalty))
			print("Avg accuracy: "  + str(epoch_accuracy))

			with torch.no_grad():
				test_outputs = mlp(test_images).flatten()
				test_acc = accuracy(test_outputs, test_labels)
				print("Test accuracy: " + str(test_acc))

	#save model for inference
	torch.save(mlp.state_dict(), filepath)

	return epoch_risk, epoch_accuracy

def test(config, test_data, filepath):
	mlp = MLP(config["dim1"], config["dim2"])
	mlp.load_state_dict(torch.load(filepath))
	
	#test on everything
	test_dataloader = DataLoader(test_data,batch_size = len(test_data))
	images, labels = next(iter(test_dataloader))

	with torch.no_grad():
		outputs = mlp(images).flatten()
		test_acc = accuracy(outputs, labels)

		print("Test accuracy: " + str(test_acc))
		return test_acc