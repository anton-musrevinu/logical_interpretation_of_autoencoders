
import z3
import itertools

def create(nbvars, nbxvars):
	if nbxvars > nbvars:
		raise Exception('The number of variables has to be larger or equal to the number of x vars')

	start_oneHot = nbxvars + 1
	variables = []
	for i in range(start_oneHot, nbvars + 1):
		variables.append(z3.Bool(str(i)))

	if len(variables) == 0:
		kb =  True
	if len(variables) == 1:
		kb =  variables[0]
	if len(variables) >= 2:
	# 	kb1 = z3.Or(variables[0],variables[1])
	# 	kb2 = z3.Or(z3.Not(variables[0]), z3.Not(variables[1]))
	# 	kb = z3.And(kb1,kb2)
	# if len(variables) >= 3:
		# kb1 = z3.Or(variables[0], variables[1], variables[2])
		# kb2 = z3.Or(z3.Not(variables[0]),z3.Not(variables[1]),z3.Not(variables[2]))
		# kb3 = z3.Or(z3.Not(variables[0]),z3.Not(variables[1]))
		# kb4 = z3.Or(z3.Not(variables[1]),z3.Not(variables[2]))
		# kb5 = z3.Or(z3.Not(variables[0]),z3.Not(variables[2]))
		# kb = z3.And(kb1,kb2,kb3,kb4,kb5)
		kbs = []
		kbs.append(z3.Or(variables))
		# kbs.append(z3.Or([z3.Not(v) for v in variables]))
		for (v1,v2) in itertools.combinations(variables, 2):
			kbs.append(z3.Or(z3.Not(v1), z3.Not(v2)))
		kb = z3.And(kbs)

	return kb

def create_all_contraints(num_fl_vars, fl_categorical_dim):
	# y_vars = list(range(num_fl_vars * fl_categorical_dim + 1, total_num_vars + 1))
	kb_vars = []

	for var in range(num_fl_vars):
		var_idx = var * fl_categorical_dim
		var_range = list(range(var_idx + 1, var_idx + fl_categorical_dim + 1))
		if var_range != []:
			kb_vars.append(create_for_list(var_range))

	# if y_vars != []:
	# 	kb_vars.append(create_for_list(y_vars))

	return z3.And(kb_vars)

def create_y_contraints(total_num_vars, num_fl_vars, fl_categorical_dim):
	y_vars = range(num_fl_vars * fl_categorical_dim + 1, total_num_vars + 1)
	return create_for_list(y_vars)


def create_for_list(vars_to_one_hot):
	print('creating onehot for vars: {}'.format(vars_to_one_hot))
	if vars_to_one_hot == []:
		raise Exception('The elements to convert to onehot have to be not empty')

	# start_oneHot = nbxvars + 1
	variables = []
	for i in vars_to_one_hot:
		variables.append(z3.Bool(str(i)))

	if len(variables) == 0:
		kb =  True
	if len(variables) == 1:
		kb =  variables[0]
	if len(variables) >= 2:
	# 	kb1 = z3.Or(variables[0],variables[1])
	# 	kb2 = z3.Or(z3.Not(variables[0]), z3.Not(variables[1]))
	# 	kb = z3.And(kb1,kb2)
	# if len(variables) >= 3:
		# kb1 = z3.Or(variables[0], variables[1], variables[2])
		# kb2 = z3.Or(z3.Not(variables[0]),z3.Not(variables[1]),z3.Not(variables[2]))
		# kb3 = z3.Or(z3.Not(variables[0]),z3.Not(variables[1]))
		# kb4 = z3.Or(z3.Not(variables[1]),z3.Not(variables[2]))
		# kb5 = z3.Or(z3.Not(variables[0]),z3.Not(variables[2]))
		# kb = z3.And(kb1,kb2,kb3,kb4,kb5)
		kbs = []
		kbs.append(z3.Or(variables))
		# kbs.append(z3.Or([z3.Not(v) for v in variables]))
		for (v1,v2) in itertools.combinations(variables, 2):
			kbs.append(z3.Or(z3.Not(v1), z3.Not(v2)))
		kb = z3.And(kbs)

	return kb