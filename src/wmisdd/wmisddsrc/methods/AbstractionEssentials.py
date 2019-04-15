import z3


def get_bound_vars_in_atom(refinement, groundAtoms = []):
	#p(refinement,type(refinement)

	if z3.is_bool(refinement):
		for elem in refinement.children():
			#print(elem,type(elem))
			get_bound_vars_in_atom(elem,groundAtoms)
	elif z3.is_rational_value(refinement) or z3.is_int_value(refinement):
		return []
	elif z3.is_arith(refinement) and len(refinement.children()) > 1:
		for elem in refinement.children():
			get_bound_vars_in_atom(elem,groundAtoms)
		#print('is arith, r: {}, parms: {}, decl: {}, num_args: {}'.format(refinement, refinement.params(), refinement.decl(), refinement.num_args()))
	elif z3.is_arith(refinement) and len(refinement.children()) == 0:
		groundAtoms.append(refinement)
		#print("heyo")
	else:
		raise AbstractionCondition("Unknown Expression: {}, {}".format(refinement,type(refinement)))

	#print('count: {}, = {}, {}'.format(refinement, count, type(refinement)))
	return groundAtoms

def mapFunctionSymbol(symbol):
	if symbol == '&':
		return _and
	elif symbol == '|':
		return _or
	elif symbol == '~':
		return _not
	elif symbol == '<=':
		return _leq
	elif symbol == '<':
		return _le
	elif symbol == '*':
		return _times
	elif symbol == '+':
		return _plus
	elif symbol == '-':
		return _minus
	elif symbol == 'ite':
		return _ite
	else:
		raise Exception("WRONG SYMBOL FOUND : {}".format(symbol))

def recusiveSimplification(formula):
	if formula.children() == 0:
		decl = formula.decl()
		children = []
		for child in formula.children():
			children.append(recusiveSimplification(z3.simplify(child)))
		if len(children) == 1:
			return decl([children[0]])
		else:
			return decl(children)
	else:
		return formula

def _ite(lst):
	if len(lst) != 3 or not z3.is_bool(lst[0]):
		raise Exception('WRONG INPUT FOR _ite: {}'.formula(lst))
	return z3.If(lst[0], lst[1], lst[2])

def _and(lst):
	return z3.And(lst)

def _or(lst):
	return z3.Or(lst)
def _not(lst):
	return z3.Not(lst)
def _le(lst):
	#self._logger.setVerbose(True)
	if len(lst) != 2:
		raise Exception('WRONG INPUT for _le: {}'.format(lst))
	# abstract = abstraction(hybridKnowlegeBase,lst[0] < lst[1],logger)
	return lst[0] < lst[1]

def _leq(lst):
	if len(lst) != 2:
		raise Exception('WRONG INPUT for _le: {}'.format(lst))
	# abstract = abstraction(hybridKnowlegeBase,lst[0] <= lst[1],logger)
	return lst[0] <= lst[1]

def _times(lst):
	if len(lst) != 2:
		raise Exception('WRONG INPUT for _times: {}'.format(lst))
	return lst[0] * lst[1]

def _plus(lst):
	result = 0
	for i in lst:
		result += i
	return result

def _minus(lst):
	result = 0
	for i in lst:
		result +- i
	return result