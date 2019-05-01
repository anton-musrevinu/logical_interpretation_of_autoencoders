import csv
import os

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
from scipy.interpolate import griddata

root = os.path.abspath('./../../')
file = os.path.join(root, 'output/var_results_mnist.csv')

names = []
fl_sizes = []
cat_sizes = []
losses  = []
model_spaces = []
model_spaces_values = {}

with open(file, 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for idx,row in enumerate(spamreader):
		if idx == 0:
			names = row
			continue
		if len(row) < 4 or (row[0].strip() == 'emnist' or '#' in row[0]):
			continue

		fl_sizes.append(int(row[1].strip()))
		cat_sizes.append(int(row[2].strip()))
		losses.append(float(row[3].strip()))
		model_idx = len(model_spaces)
		model_spaces.append(model_idx)
		model_spaces_values[model_idx] = row[4].strip()

# create x-y points to be used in heatmap
xi = np.sort(np.unique(fl_sizes))
yi = np.sort(np.unique(cat_sizes))

# Z is a matrix of x-y values
zi = griddata((fl_sizes, cat_sizes), losses, (xi[None,:], yi[:,None]), method='cubic')
model_space = griddata((fl_sizes, cat_sizes), model_spaces, (xi[None,:], yi[:,None]), method='cubic')

print(xi)
print(yi)
print(zi)
print(model_space)
# Create the contour plot
# CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
				  # vmax=max(losses), vmin=min(losses))
fig_1 = plt.figure(figsize=(8, 4))
ax = fig_1.add_subplot(111)

extent = [xi[0], xi[-1], yi[0], yi[-1]]
img = ax.imshow(zi, interpolation='nearest')#,extent=extent)
plt.colorbar(img)
# We want to show all ticks...
ax.set_xticks(np.arange(len(xi)))
ax.set_yticks(np.arange(len(yi)))
# ... and label them with the respective list entries
ax.set_xticklabels(xi)
ax.set_yticklabels(yi)

offsetx = 0.25
offsety = -0.3

for i in range(len(yi)):
	for j in range(len(xi)):
		idx = model_space[i, j]
		text = ax.text(j + offsetx, i + offsety, model_spaces_values[idx], ha="center", va="center", color="black")

ax.grid(False)

ax.set_title("mean squared (reconstuction) error for vae on mnist")
ax.set_xlabel('# variables in the FL')
ax.set_ylabel('categorical dim of FL variables')
fig_1.savefig('./vae_results_mnist.pdf')
plt.show()






















