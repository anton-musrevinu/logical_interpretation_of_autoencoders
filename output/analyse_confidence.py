
names = []
file = {}
with open('./confidence.csv', 'r') as f:
	for line_idx, line in enumerate(f):
		print(line)
		line_split = line.replace('\n','').split(',')
		if line_idx == 0:
			for i in line_split:
				i = i.strip()
				file[i] = []
				names.append(i)
		else:
			for idx, i in enumerate(line_split):
				file[names[idx]].append(i)

# for i in range(len(file['task'])):
# 	possible_all = (float(file['possible_a'][i]) + float(file['possible_b'][i]))/2
# 	impossible   = float(file['impossible'][i])

# 	possible_norm = possible_all / (possible_all + impossible)
# 	impossible_norm = impossible / (possible_all + impossible)

# 	print('task: {} \t possible: {}, \t impossible: {}\t, norm: {} vs {}'.format(file['task'][i], possible_all, impossible, possible_norm, impossible_norm))


with open('./confidence_out.csv', 'w') as f:
	f.write('task, possible-[y=0], possible-[y=1],possible, impossible\n')
	for i in range(len(file['task'])):
		possible_a = float(file['possible_a'][i])
		possible_b = float(file['possible_b'][i])
		impossible  = float(file['impossible'][i])

		possible_a_norm = possible_a / (possible_a + possible_b + impossible)
		possible_b_norm = possible_b / (possible_a + possible_b + impossible)
		impossible_norm = impossible / (possible_a + possible_b + impossible)

		possible_norm = (possible_a +  possible_b)/ (possible_a + possible_b + impossible)

		print('task: {} \t norm: possible-[y=0]: {} possible-[y=1]: {} vs impossible {}'.format(file['task'][i], possible_a_norm, possible_b_norm, impossible_norm))

		f.write('{},{:.4},{:.4},{:.4},{:.4}\n'.format(file['task'][i], possible_a_norm, possible_b_norm, possible_norm, impossible_norm))