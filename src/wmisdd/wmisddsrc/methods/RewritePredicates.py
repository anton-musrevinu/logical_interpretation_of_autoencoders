
from ..elements.Function import WeightFunction
from ..elements.myExceptions import RewritingError
from ..elements.Bound import Bound

import z3
import importlib
import functools

def construct_variable_order(weightFunction, orderType = WeightFunction.DEF_VAR_ORDER_ALPH, hybridKnowlegeBase = None):
	boundRealVars, boundBoolVars = weightFunction.get_bound_variables()
	orderReal = _construct_order_alp(boundRealVars)
	orderBool = _construct_order_alp(boundBoolVars)

	return orderReal, orderBool

def _construct_order_alp(variableList):
	orderVariableList = sorted(variableList, key = lambda t: str(t), reverse = False)
	return orderVariableList

def construct_order(wf,hybridKnowlegeBase):
	refVrasDict = {}
	max_var_value = 0
	max_var = None

	for key, values in hybridKnowlegeBase.get_ground_var_refs().items():
		refvalues = []
		for x in values:
			refvalues = refvalues + (hybridKnowlegeBase.get_predicate(x).get_bound_variables())
		refvalues = list(set(refvalues))
		refVrasDict[key] = len(refvalues)

	o = OrderedDict(sorted(refVrasDict.items(), key=lambda t: t[1],reverse = True))

	varorder = {}
	for key, value in o.items():
		varorder[len(varorder)] = key

	return varorder

def rewrite_all_predicates(hybridKnowlegeBase, orderedRealVars, logger):

	for predicate in hybridKnowlegeBase.get_predicates():

		#Skipping all predicates representing simple boolean variables
		if not predicate.hasRefinement:
			logger.writeToLog('P{}\t: formula: {}'.format(predicate.id, predicate.formula),'info')
			continue

		initialized = False
		res = []
		for i, var in enumerate(orderedRealVars):
			if var in predicate.get_bound_variables() and not initialized:
				oldFormula = predicate.get_fomula()
				newFormula, bound = init_predicate_bound(oldFormula,var,logger)
				predicate.set_leadVar(var)
				predicate.set_newFormula(newFormula)
				predicate.set_bound(bound)
				initialized = True
				logger.writeToLog('P{}\t: original: {}, new: {}, Bound: {}, lead: {}, boundVars: {}'\
					.format(predicate.id, oldFormula, newFormula, bound,var, predicate.get_bound_variables()), 'info')
				break
		if not initialized:
			raise RewritingError('The refinement of the predicate mentions variables that are not referenced in the Varibale Order')

def init_predicate_bound(oldFomula, leadVar,logger):
	try:
		newFormula = rearage_formula(oldFomula, leadVar)
	except Exception as e:
		# logger.error(e)
		print('rearage_formula failed on formula: {} and leadVar: {}'.format(oldFomula,leadVar),'error')
		raise e

	#in newFormula the leadVar should be on one side of the inquality alone!
	leadVarIsLeft = newFormula.children()[0] == leadVar
	op = newFormula.decl().kind()
	if op == z3.Z3_OP_LE or op == z3.Z3_OP_LT:
		if leadVarIsLeft:
			bound = Bound(Bound.TYPE_UPPER, newFormula.children()[1])
		else:
			bound = Bound(Bound.TYPE_LOWER, newFormula.children()[0])
	elif op == z3.Z3_OP_GE or op == z3.Z3_OP_GT:
		if leadVarIsLeft:
			bound = Bound(Bound.TYPE_LOWER, newFormula.children()[1])
		else:
			bound = Bound(Bound.TYPE_UPPER, newFormula.children()[0])

	return newFormula, bound

def rearage_formula(formula, leadVar):
	#This method takes a z3 arithmetic inequality and returns an inequality where the leadVar is alone on one site

	leftPart = formula.children()[0]
	rightPart = formula.children()[1]
	leadVarInLeft = leadVar in get_bound_vars_in_atom(leftPart,[])
	leadVarInRight = leadVar in get_bound_vars_in_atom(rightPart,[])
	if leadVarInRight and leadVarInLeft:
		raise Exception('leadVar appears in both sides of the inequality, this is not yet implemented')
	elif leadVarInLeft and not leadVarInRight:
		aim = leftPart
		rest = rightPart
	elif not leadVarInLeft and leadVarInRight:
		aim = rightPart
		rest = leftPart
	else:
		raise Exception('leading variable does not appear in any part of the inequality')

	if aim == leadVar:
		return formula

	for elem in aim.children():
		boundInElem = get_bound_vars_in_atom(elem, [])
		if not leadVar in boundInElem:
			moveElem = elem
			break

	if z3.is_add(aim):
		newLeftPart = z3.simplify(leftPart - moveElem)
		newRightPart = z3.simplify(rightPart - moveElem)
		newFormula = formula.decl()(newLeftPart, newRightPart)
		return rearage_formula(newFormula, leadVar)

	elif z3.is_sub(aim):
		newLeftPart = z3.simplify(leftPart + moveElem)
		newRightPart = z3.simplify(rightPart + moveElem)
		newFormula = formula.decl()(newLeftPart, newRightPart)
		return rearage_formula(newFormula, leadVar)

	elif z3.is_mul(aim) and (z3.is_rational_value(moveElem) or z3.is_int_value(moveElem)):
		newLeftPart = z3.simplify(leftPart * 1/moveElem)
		newRightPart = z3.simplify(rightPart * 1/moveElem)
		if float(str(moveElem)) > 0:
			newFormula = formula.decl()(newLeftPart, newRightPart)
		else:
			#Multiplying by a negative number we have to change the inequality around
			newFormula = formula.decl()(newRightPart, newLeftPart)
		return rearage_formula(newFormula, leadVar)

	elif z3.is_div(aim) and (z3.is_rational_value(moveElem) or z3.is_int_value(moveElem)):
		newLeftPart = z3.simplify(leftPart * moveElem)
		newRightPart = z3.simplify(rightPart * moveElem)
		if float(str(moveElem)) > 0:
			newFormula = formula.decl()(newLeftPart, newRightPart)
		else:
			#Multiplying by a negative number we have to change the inequality around
			newFormula = formula.decl()(newRightPart, newLeftPart)
		return rearage_formula(newFormula, leadVar)

	elif z3.is_mul(aim) and not (z3.is_rational_value(moveElem) or z3.is_int_value(moveElem)):
		# print('aim',aim)
		# print('moveElem',moveElem)
		# print('leftPart', leftPart)
		# print('rightPart', rightPart)
		# print('formula', formula)
		# print('leadVar', leadVar)
		newLeftPart = z3.simplify(devide_by_variable(leftPart, moveElem))
		# newLeftPart = z3.simplify(leftPart * 1/moveElem)
		newRightPart = z3.simplify(devide_by_variable(rightPart, moveElem))
		newFormula = formula.decl()(newLeftPart, newRightPart)
		# print('deviding by variable', aim, formula)
		# print('left:', leftPart, newLeftPart)
		# print('right:', rightPart, newRightPart)
		return rearage_formula(newFormula, leadVar)
		# raise Exception('Trying to devide formula by variable, this is not yet implemented')
	else:
		raise Exception('Unkown function operation has been detected for rearanging this element: {}, {}, leadVar: {}'.format(formula, moveElem, leadVar))

def devide_by_variable(part, moveElem):
	if moveElem in part.children() and z3.is_mul(part):
		children = part.children()[:]
		del children[children.index(moveElem)]
		result = functools.reduce(lambda x, y: x*y, children)
		return result
	else:
		return part * 1/moveElem


def _condition_only_number(elem, varsInElem):
	aimSim = z3.simplify(elem)
	cond1 = varsInElem == [] \
			and (z3.is_rational_value(aimSim) or z3.is_int_value(aimSim))

	return cond1

def _condition_sum_of_var_times_coef(rest,varsInRest):
	cond2 = True
	cond2 = cond2 and varsInRest != []

	if z3.is_add(rest):
		cond2 = cond2 and len(rest.children()) == len(varsInRest)
	elif rest.decl().kind() == z3.Z3_OP_UMINUS:
		cond2 = cond2 and len(rest.children()) == 1
		cond2 = cond2 and z3.is_const(rest.children()[0])
	elif rest.children() == []:
		cond2 = cond2 and len(varsInRest) == 1
		cond2 = cond2 and z3.is_const(rest)
	elif z3.is_mul(rest):
		cond2 = cond2 and len(rest.children()) == 2
		elem = rest.children()[0]
		varsInElem = get_bound_vars_in_atom(elem,[])
		cond2 = cond2 and _condition_only_number(elem, varsInElem)
		cond2 = cond2 and z3.is_const(rest.children()[1])
	else:
		cond2 = cond2 and False

	return cond2


def rearage_formula_latte(formula, logger):
	#This method takes a z3 arithmetic inequality and returns an inequality where the number is alone on one site

	formula = z3.simplify(formula, arith_lhs = True)

	leftPart = z3.simplify(formula.children()[0])
	rightPart = z3.simplify(formula.children()[1])

	logger.writeToLog('formula: {}, leftPart: {}, rightPart: {}'.format(formula, leftPart, rightPart),'debug')

	aim = rightPart
	rest = leftPart

	logger.writeToLog('aim: {}, rest: {}'.format(aim, rest))

	varsInAim = get_bound_vars_in_atom(aim,[])
	varsInRest = get_bound_vars_in_atom(rest,[])

	logger.writeToLog('varsInAim: {}, varsInRest: {}'.format(varsInAim, varsInRest))

	cond1 = _condition_only_number(aim, varsInAim)
	cond2 = _condition_sum_of_var_times_coef(rest, varsInRest)
	logger.writeToLog('cond1: {}, cond2: {}'.format(cond1, cond2))

	if cond1 and cond2:
		t = create_inequality(aim,rest, formula.decl())
		return t
	else:
		raise Exception('something wrong here: {}'.format(formula))


def create_inequality(aim,rest, decl):
	op = decl.kind()
	if op == z3.Z3_OP_LE or op == z3.Z3_OP_LT:
		return decl(rest,aim)
	elif op == z3.Z3_OP_GE:
		t = z3.simplify(-rest, som = True)
		p = z3.simplify(-aim, som = True)
		return t <= p
	elif op == z3.Z3_OP_GT:
		return z3.simplify(-rest, som = True) < z3.simplify(-aim, som = True)

def get_bound_vars_in_atom(refinement, groundAtoms = []):
	#print(refinement,type(refinement)

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
	elif z3.is_arith(refinement) and len(refinement.children()) == 1 and refinement.decl().kind() == z3.Z3_OP_UMINUS:
		get_bound_vars_in_atom(refinement.children()[0], groundAtoms)
		#print("heyo")
	else:
		raise RewritingError("Unknown Expression: {}, {}".format(refinement,type(refinement)))

	#print('count: {}, = {}, {}'.format(refinement, count, type(refinement)))
	return groundAtoms
