import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import torch
import torch.nn.functional as F
import numpy as np
from src.pytorch_experiment_scripts.model_architectures import DoubleLeakyReLU

def plot_tanh(precision = 10**(-2), range = [-10,10]):
	fig_1 = plt.figure(figsize=(8, 4))
	ax_1 = fig_1.add_subplot(111)
	x_values = torch.arange(range[0], range[1], precision, dtype = torch.float)
	y_values = torch.tanh(x_values)
	y_values_2 = torch.tanh(x_values * 2)
	x_values_np = x_values.data.numpy()
	ax_1.plot(x_values_np, y_values.data.numpy())
	ax_1.plot(x_values_np, y_values_2.data.numpy())
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape))
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape) + 1)
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape) - 1)
	plt.show()

def plot_hardtanh(precision = 10**(-2), range = [-10,10]):
	fig_1 = plt.figure(figsize=(8, 4))
	ax_1 = fig_1.add_subplot(111)
	x_values = torch.arange(range[0], range[1], precision, dtype = torch.float)
	y_values = F.hardtanh(x_values, min_val = -1, max_val = 1)
	print(x_values)
	print(y_values)
	x_values_np = x_values.data.numpy()
	ax_1.plot(x_values_np, y_values.data.numpy())
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape))
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape) + 1)
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape) - 1)
	plt.show()

def plot_DoubleLeakyRelu(precision = 10**(-4), range = [-3,3], t = None):
	doubleLeakyRelu = DoubleLeakyReLU(slope = 1,breakpoint = 1, leaky_slope = .1)
	fig_1 = plt.figure(figsize=(8, 4))
	ax_1 = fig_1.add_subplot(111)
	x_values = torch.arange(range[0], range[1], precision, dtype = torch.float)
	if t != None:
		y_values = t.forward(x_values)
	else:
		y_values = doubleLeakyRelu.forward(x_values)
	# print(x_values)
	# print(y_values)
	x_values_np = x_values.data.numpy()
	ax_1.plot(x_values_np, y_values.data.numpy())
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape))
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape) + 1)
	ax_1.plot(x_values_np, np.zeros(x_values_np.shape) - 1)
	plt.show()

def test():
	weight_decay_coefficient = 1e-05
	# model = torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),torch.nn.ReLU())
	t = DoubleLeakyReLU(slope = 2,breakpoint = 1, leaky_slope = 0.05)
	model = torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),t)
	criterion = torch.nn.MSELoss()
	print(list(model.parameters()))
	# optimizer = torch.optim.Adam(model.parameters(), amsgrad=False,
									# weight_decay=weight_decay_coefficient)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	n_in, n_h, n_out, batch_size = 2, 100000, 2, 100
	x_all = torch.randn(n_h, n_in)
	bias = 100
	y_all = torch.from_numpy(np.heaviside(x_all.data.numpy(), 1) * 2 -1)

	# print(x)
	# print(y)
	# print(model.parameters())
	print(x_all)

	for epoch in range(20):

		perm = np.random.RandomState(7112018).permutation(x_all.shape[0])
		x_all = x_all[perm]
		y_all = y_all[perm]
		# Forward Propagation
		for i in range(int(n_h/batch_size)): 
			x = x_all[i * batch_size:(i+1) * batch_size]
			# print(x.shape,x)
			y = y_all[i * batch_size:(i+1) * batch_size]

			y_pred = model(x)


			# Compute and print loss
			loss = criterion(y_pred, y)

			# Zero the gradients
			if i < int(n_h/batch_size) -1:
				optimizer.zero_grad()
				
				# perform a backward pass (backpropagation)
				loss.backward()
				
				# Update the parameters
				optimizer.step()

		print('epoch: {}. x: {}, predicted y: {}, actual y: {}, loss: {}'.format(epoch,x[0],y_pred[0], y[0], loss.item()))

	print(list(model.parameters()))
	plot_DoubleLeakyRelu(t = t)

if __name__ == "__main__":

	# compare_test_acc_val()
	# test() 
	plot_tanh()






