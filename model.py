import torch

#randomizing this split?
def penalty(outputs, labels):
	w = torch.tensor(1).float().requires_grad_()
	env_loss1 = risk(w*outputs[0::2], labels[0::2])
	env_loss2 = risk(w*outputs[1::2],labels[1::2])
	gradient1 = torch.autograd.grad(env_loss1, w, create_graph = True)[0]
	gradient2 = torch.autograd.grad(env_loss2, w, create_graph=True)[0]
	#estimate of squared L2 norm but "gradient" is just a single value
	return gradient1*gradient2


def risk(outputs, labels):
	return torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

def accuracy(outputs,labels):
	predicted_labels = (outputs>0).float()
	return (predicted_labels == labels).float().mean()

def init_weights(m):
	if type(m)==torch.nn.Linear:
		torch.nn.init.kaiming_uniform_(m.weight)
		torch.nn.init.zeros_(m.bias)

class MLP(torch.nn.Module):
	def __init__(self, dim1, dim2, input_dim = 3*14*14):
		super(MLP, self).__init__()
		self.lin1 = torch.nn.Linear(input_dim, dim1)
		self.lin2 = torch.nn.Linear(dim1,dim2)
		self.lin3 = torch.nn.Linear(dim2,1)
		self.activation = torch.nn.ReLU()

		self.apply(init_weights)

	def forward(self, x):
		x = x.view(x.shape[0],-1)
		x = self.lin1(x)
		x = self.activation(x)
		x = self.lin2(x)
		x = self.activation(x)
		x = self.lin3(x)
		return x
