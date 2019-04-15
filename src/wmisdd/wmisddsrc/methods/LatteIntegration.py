from .RewritePredicates import rearage_formula_latte
from .IntegrationEssentials import create_real_interval
from ..elements.myExceptions import LatteException
from ..elements.Interval import Interval
import z3
from BitVector import BitVector
import numpy as np
import itertools
import time
from subprocess import call
import multiprocessing as mp
import os

POLYTOPE_TEMPLATE = 'polytop.hrep.latte'
ALG_TRIANGULATE = "--triangulate"
ALG_CONE_DECOMPOSE = "--cone-decompose"

def write_integration_problem_to_latte_file(intervalsReal, tmpDir, model, orderedReals,logger):
	problemDir = tmpDir + '{}/'.format(model)
	os.mkdir(problemDir)
	file = problemDir + POLYTOPE_TEMPLATE
	allmats = [_get_mat_from_interval(interval, orderedReals,logger) for interval in intervalsReal]

	aaMat = allmats[0]

	for i in allmats[1::]:
		aaMat = np.concatenate((aaMat, i), axis = 0)
	if len(aaMat.shape) == 1:
		(mm, ddp1) = (len(mm), 1)
	else:
		(mm, ddp1) = aaMat.shape
	with open(file, 'w') as f:
		f.write('{} {}\n'.format(mm,ddp1))
		for row in aaMat:
			rowString = ''
			for elem in row:
				elem = str(elem)
				if '.' in elem and len(elem.split('.')) == 2:
					elem = elem.split('.')[0]
				rowString = rowString + elem + ' '
			# logger.writeToLog('writing row: {}'.format(rowString),'debug')
			f.write(rowString + '\n')

	return problemDir

def _get_mat_from_interval(interval, orderedReals,logger):
	formulas = list(itertools.chain(_get_upper_bound_from_interval(interval,logger),_get_lower_bound_from_interval(interval,logger)))
	dd = len(orderedReals)
	mm = len(formulas)

	aab = np.zeros((mm,dd + 1))

	for ii,formula in enumerate(formulas):
		if z3.is_le(formula):
			aa = formula.children()[0]
			bb = formula.children()[1]
		elif z3.is_ge(formula):
			aa = formula.children()[1]
			bb = formula.children()[0]

		logger.writeToLog('aa: {}, bb: {}'.format(aa,bb),'debug')
		aab[ii][0] = str(int(str(bb)))
		if z3.is_add(aa):
			for elem in aa.children():
				pos, coef = get_moninomial(elem, orderedReals,logger)
				elem = str(z3.simplify(-coef))
				if '.' in elem and len(elem.split('.')) == 2:
					elem = elem.split('.')[0]
				aab[ii][pos + 1] = elem
		else:
			pos, coef = get_moninomial(aa, orderedReals,logger)
			elem = str(z3.simplify(-coef))
			if '.' in elem and len(elem.split('.')) == 2:
				elem = elem.split('.')[0]
			aab[ii][pos + 1] = elem
	return aab

def get_moninomial(elem, orderedReals, logger):
	if elem.decl().kind() == z3.Z3_OP_MUL:
		coef = elem.children()[0]
		var = elem.children()[1]
	elif z3.is_const(elem):
		coef = z3.IntVal(1)
		var = elem
	else:
		raise Exception('Unknonwn moninoial: {}'.format(elem))

	pos = orderedReals.index(var)
	# logger.writeToLog('var: {}, pos: {}, coef: {}'.format(var, pos, coef),'debug')
	# print(coef)
	return pos, coef


def _get_upper_bound_from_interval(interval,logger):
	bound = interval.get_upper_bound()
	leadVar = interval.get_leadVar()

	funclist = bound.get_func_list()[:]
	if bound.has_float():
		funclist.append(bound.get_float())

	for func in funclist:
		formula = (leadVar <= func)
		yield rearage_formula_latte(formula, logger)

def _get_lower_bound_from_interval(interval,logger):
	bound = interval.get_lower_bound()
	leadVar = interval.get_leadVar()

	funclist = bound.get_func_list()[:]
	if bound.has_float():
		funclist.append(bound.get_float())

	for func in funclist:
		formula = (leadVar >= func)
		yield rearage_formula_latte(formula,logger)

def create_latte_file_for_model_new(model, hybridKnowlegeBase, orderedReals, tmpDir, logger):
	logger.writeToLog(" -- -- Creating Real Intervals for model - real mask: {} ".format(model)+ \
		'with bound variables: {} '.format(hybridKnowlegeBase.get_ground_var_refs()),'debug')

	intervalsReal = []
	for leadVar in orderedReals:
		referencedPredicats = hybridKnowlegeBase.get_referenced_predicates(leadVar)
		for predicate in referencedPredicats:
			assignment = model[predicate.get_id() - 1] == 1
			interval = Interval(leadVar)
			interval.combine_bound(predicate.get_bound(), assignment)
			intervalsReal.append(interval)

	file = write_integration_problem_to_latte_file(intervalsReal, tmpDir, model, orderedReals,logger)

	# logger.writeToLog('Writing intervals for filter {} to file => {}\n\t'.format(model,file)\
	# 	 + str([str(interval) for interval in intervalsReal]),'debug')

	return file

def create_latte_file_for_model(model, hybridKnowlegeBase, orderedReals, tmpDir, logger):

	logger.writeToLog(" -- -- Creating Real Intervals for model - real mask: {} ".format(model)+ \
		'with bound variables: {} '.format(hybridKnowlegeBase.get_ground_var_refs()),'debug')

	intervalsReal = []
	for leadVar in orderedReals:
		referencedPredicats = hybridKnowlegeBase.get_referenced_predicates(leadVar)
		interval = create_real_interval(leadVar, model, referencedPredicats, logger)
		if interval == False:
			return False
		intervalsReal.append(interval)

	file = write_integration_problem_to_latte_file(intervalsReal, tmpDir, model, orderedReals,logger)

	# logger.writeToLog('Writing intervals for filter {} to file => {}\n\t'.format(model,file)\
	# 	 + str([str(interval) for interval in intervalsReal]),'debug')

	return file

def init_latte_weight_function(wf, tmpDir, logger):
	wfAsLatteList = convert_z3_func_to_latte(wf)

	tmpFile = tmpDir + 'tmp_weight_function.polynomial.latte'

	write_latte_to_file(wfAsLatteList, tmpFile)
	return tmpFile

def write_latte_to_file(wfAsLatteList, tmpFile):
	with open(tmpFile,'w') as f:
		f.write(str(wfAsLatteList))

def _convert_singleton_to_list(monomial, varOrder):
	exponentVector = np.zeros(len(varOrder))
	if z3.is_rational_value(monomial) or z3.is_int_value(monomial):
			#Constant Function such as (10)
		return monomial, exponentVector
	elif z3.is_const(monomial):
			#Individual Variable eg. (x)
		exponentVector[varOrder.index(monomial)] = 1
		return 0, exponentVector
	elif monomial.decl().kind() == 529:
		ex = monomial.children()[1]
		if not (z3.is_rational_value(ex) or z3.is_int_value(ex)):
			raise Exception('exponetion is not numeric: {}'.format(ex))
		var = monomial.children()[0]
		exponentVector[varOrder.index(var)] = float(str(ex))
		return 0, exponentVector
	else:
		raise Exception('non singleton trying to be converted {}'.format(monomial))

def _convert_monomial_to_list(monomial, varOrder):
	exponentVector = np.zeros(len(varOrder))
	coefficient = 0
	if z3.is_mul(monomial):
		for singleton in monomial.children():
			[coef, expnVec] = _convert_monomial_to_list(singleton,varOrder)
			coefficient += int(str(coef))
			exponentVector = exponentVector + np.array(expnVec)
			#print(singleton,'adding', coef, expnVec, 'res:', coefficient, exponentVector)
	else:
		coef, expnVec = _convert_singleton_to_list(monomial,varOrder)
		coefficient += int(str(coef))
		exponentVector = exponentVector + expnVec

	return [coefficient, [int(x) for x in exponentVector]]


def convert_z3_func_to_latte(wf):
	wfnew = z3.simplify(wf.get_function_as_z3(), som = True)
	varOrder = wf.get_real_variable_order()
	z3inLatte = []
	if z3.is_add(wfnew):
		#Product of Polynomials (x + y)
		for monomial in wfnew.children():
			[coef, expVec] = _convert_monomial_to_list(monomial, varOrder)
			coef = 1 if coef == 0 else coef
			z3inLatte.append([coef, expVec])

	else:
		[coef, expVec] = _convert_monomial_to_list(wfnew, varOrder)
		coef = 1 if coef == 0 else coef
		z3inLatte.append([coef, expVec])

	return z3inLatte

def compute_latte_integration_for_intervals(weightFunctionFile, polytopeDIR, intervalsBool, nbInt, integrationProblemID,tmpDir, logger):
	start_time = time.time()
	os.chdir(polytopeDIR)
	polytopeFile = polytopeDIR + POLYTOPE_TEMPLATE
	outputFile = polytopeDIR + 'polytop_{}.out'.format(integrationProblemID)
	try:
		_do_latte_integartion(weightFunctionFile, polytopeFile, outputFile)
	except Exception as e:
		logger.error('integration error: {} from file: {}'.format(e, polytopeFile))
		return None

	intTime = time.time() - start_time

	vol = _read_output_file(outputFile)
	if vol == None:
		out = ''
		with open(outputFile, 'r') as f:
			for line in f:
				out = out + '\n' +  line
		# logger.writeToLog(out,'result')
		vol = 0

	threadId =  mp.current_process()

	if vol == 0:
		logger.writeToLog(' == Integrate: ({},{:.2}, {}) x {} ==\n\t== {} || {} || {} on thread: {}'.format(vol,intTime,integrationProblemID, nbInt,\
		weightFunctionFile,polytopeFile, intervalsBool, threadId),'info')
	else:
		logger.writeToLog(' == Integrate: ({:.3},{:.2}, {}) x {} ==\n\t== {} || {} || {} on thread: {}'.format(vol,intTime,integrationProblemID, nbInt,\
			weightFunctionFile,polytopeFile, intervalsBool, threadId),'info')

	vol = vol * nbInt

	return (vol, 0)

def _do_latte_integartion(polynomial_file, polytope_file, output_file):
	with open(output_file,'w') as f:
		return_value = call(["integrate",
							 "--valuation=integrate", ALG_CONE_DECOMPOSE,
							 "--monomials=" + polynomial_file,
							 polytope_file], stdout=f, stderr=f)
		if return_value != 0:
			msg = "LattE returned with status {}"

def _read_output_file(path):
	with open(path, 'r') as f:
		for line in f:
			# result in the "Answer" line may be written in fraction form
			if "Decimal" in line:
				return float(line.partition(": ")[-1].strip())
	return None
