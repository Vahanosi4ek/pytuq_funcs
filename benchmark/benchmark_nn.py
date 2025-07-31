from benchmark import *

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pytuq.utils.plotting import *

myrc()

class MLP(nn.Module):
	def __init__(self, inputs, hidden_1, hidden_2, outputs):
		super().__init__()
		self.inputs = inputs
		self.hidden_1 = hidden_1
		self.hidden_2 = hidden_2
		self.outputs = outputs

		self.model = nn.Sequential(
			nn.Linear(self.inputs, self.hidden_1),
			nn.Sigmoid(),
			nn.Linear(self.hidden_1, self.hidden_2),
			nn.Sigmoid(),
			nn.Linear(self.hidden_2, self.outputs),
		)

	def forward(self, x):
		return self.model(x)

class NNWrapper(Function):
	def __init__(self, nnet, name="NNWrapper"):
		super().__init__()
		self.nnet = nnet

	def __call__(self, x):
		return self.nnet(torch.from_numpy(x)).float().detach().numpy()

def train(model, X_train, y_train, criterion, optimizer, epochs=1000):
	model.train()

	X_train = torch.from_numpy(X_train).float()
	y_train = torch.from_numpy(y_train).float()

	for epoch in range(epochs):
		predicted = model(X_train)
		loss = criterion(predicted, y_train)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (epoch + 1) % 1000 == 0:
			print(f"Epoch [{epoch + 1}/{epochs}], Loss {loss.item():.4f}")

def get_sample_points(func, n_points=100, data_noise=0.1):
	x = func.sample_uniform(n_points)
	y = torch.from_numpy(func(x) + np.random.randn(*x.shape) * data_noise).detach().numpy()

	return x, y

def plt_sample_points(func, x, y):
	plt.plot(x, y, "go", zorder=1000)

def plt_model(func, model, x_train, y_train):
	x = torch.from_numpy(np.linspace(func.domain[0, 0], func.domain[0, 1], 1000)).float().reshape(-1, 1)
	y = model(x).detach().numpy()

	plt_sample_points(func, x_train, y_train)
	plt.plot(x, y, "b--", zorder=2000)
	plt_func(func, ax=plt.gca())

def eval_model(func, model, criterion, n_points=100):
	model.eval()	

	x = torch.from_numpy(func.sample_uniform(n_points)).float()
	y = func(x)
	predictions = model(x)

	loss = criterion(predictions, y)

	return loss

def plt_func(func, ax=None):
	name = f"figs/{func.name}_model.png"
	plot_1d(func, func.domain, ax=ax, figname=name)
