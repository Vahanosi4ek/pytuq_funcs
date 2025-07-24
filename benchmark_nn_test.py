from benchmark_nn import *


funcs_to_train = [
	Forrester(),
	Sine1d(),
	GramacyLee2(),
]

parameters = [
	[0.4, 2000],
	[0.1, 1000],
	[0.1, 10000]
]

for func, param in zip(funcs_to_train, parameters):
	noise = param[0]
	epochs = param[1]

	print(f"===== {func.name} =====")
	model = MLP(func.dim, 128, 128, func.outdim)

	x, y = get_sample_points(func, data_noise=noise)
	plt_sample_points(func, x, y)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

	train(model, x, y, criterion, optimizer, epochs=epochs)
	loss = eval_model(func, model, criterion)
	print(f"Validation MSE loss: {loss:.4f}")

	plt_model(func, model, x, y)

	plt.clf()