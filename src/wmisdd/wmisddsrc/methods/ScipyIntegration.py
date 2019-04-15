from .IntegrationEssentials import create_real_interval
import time
from scipy import integrate
import os
import multiprocessing as mp
import importlib


def init_scipy_weight_function(weightFunction,tmpDir,logger):

	functionAsPythonString, identifier = convert_to_python_func(weightFunction.get_function_as_z3(),\
					weightFunction.get_real_variable_order(), weightFunction.get_bool_variable_order(), logger)	

	tmpFile = tmpDir + 'tmp_weight_function.py'
	_parse_write(functionAsPythonString, tmpFile)
	weightFunction.set_func_and_identifier(None, identifier)

	return tmpFile

def _parse_write(functionAsPythonString, tmpFile):
	with open(tmpFile,'w') as f:
		for line in functionAsPythonString:
			f.write(line)

def create_function_body_string(weightFunctionZ3):
	if weightFunctionZ3.children() != []:
		children = [str(create_function_body_string(child)) for child in weightFunctionZ3.children()]
		return _createFunctionForSymbol(str(weightFunctionZ3.decl()),children)
	return str(weightFunctionZ3)

def _createFunctionForSymbol(funcSym,variables):
	arithmeticFucntions = "* + And Or - <= >="
	singletonFunctions = 'Not'
	doubleFunctions = '**'
	ifelseStatement = 'If'
	if funcSym in arithmeticFucntions:
		term = '( ' + variables[0] + ' '
		for var in variables[1::]:
			term += _mapFunctionSymbol(funcSym) + ' ' + var + ' '
		term += ")"
	elif funcSym in singletonFunctions and len(variables) == 1:
		term = '{}({})'.format(_mapFunctionSymbol(funcSym),variables[0])
	elif funcSym in doubleFunctions and len(variables) == 2:
		term = '({}{}{})'.format(variables[0],_mapFunctionSymbol(funcSym),variables[1])
	elif funcSym in ifelseStatement and len(variables) == 3:
		term = '({} if {} else {})'.format(variables[1], variables[0],variables[2])
	else:
		raise Exception("WRONG SYMBOL FOUND : {}, len: {}, type: {}".format(funcSym,len(variables), type(variables)))
	return term

def _mapFunctionSymbol(symbol):
	if symbol == 'And':
		return 'and'
	elif symbol == 'Or':
		return 'or'
	elif symbol == 'Not':
		return 'not'
	elif symbol == '**':
		return '**'
	elif symbol in '- * + < <= > >=':
		return symbol
	else:
		raise Exception("WRONG SYMBOL FOUND : {}".format(symbol))

def convert_to_python_func(weightFunctionZ3, orderedRealVars, orderedBoolVars, logger):
	functionBodyString = '\treturn {}\n'.format(create_function_body_string(weightFunctionZ3))
	functionHeadString, identifier = create_function_head_string(orderedRealVars, orderedBoolVars, logger)

	functionAsString = functionHeadString + functionBodyString
	return functionAsString, identifier

def create_function_head_string(orderedRealVars, orderedBoolVars,logger):

	# # body = '\treturn {}\n'.format(functionBodyString)
	# logger.writeToLog('Body of the Function: {}'.format(body))

	head = 'def trueWeightFunc('
	elementsInHead = 0
	for i, var in enumerate(orderedRealVars):
		strvar = str(var)
		if i == 0:
			head += '{}'.format(strvar)
			elementsInHead += 1
		else:
			head += ',{}'.format(strvar)
			elementsInHead += 1

	for i, var in enumerate(orderedBoolVars):
		if elementsInHead == 0:
			head += '{}'.format(var)
		else:
			head += ',{}'.format(var)
	head += '):\n'

	identifier = head.replace('def trueWeightFunc','wf').replace('\n','')
	logger.writeToLog('Init of the function: {}'.format(head))

	return head, identifier



def create_scipy_file_for_model(model, hybridKnowlegeBase, orderedReals,orderedBools,tmpDir, logger):

	logger.writeToLog(" -- -- Creating Real Intervals for model - real mask: {} ".format(model)+ \
		'with bound variables: {} '.format(hybridKnowlegeBase.get_ground_var_refs()),'debug')

	intervalsReal = []
	for leadVar in orderedReals:
		referencedPredicats = hybridKnowlegeBase.get_referenced_predicates(leadVar)
		interval = create_real_interval(leadVar, model, referencedPredicats, logger)
		if interval == False:
			return False
		intervalsReal.append(interval)

	for i, leadVar in enumerate(orderedReals):
		varsToRefInFunc = orderedReals[(i + 1):] + orderedBools
		interval = intervalsReal[i]
		if not interval.get_leadVar() == leadVar:
			raise Exception('the interval does not math up with the imposed order: leadVar: {}, interval.leadVar: {}'.format(leadVar, interval.get_leadVar()))

		interval.init_function(i, varsToRefInFunc, logger)

		# + str(model) + ',\toder: {}\t, intervals: {}'.format(\
		# 		orderedReals, str([str(interval) for interval in intervalsReal])),'info')

	file = write_integration_problem_to_file(intervalsReal, tmpDir, model)

	logger.writeToLog('Writing intervals for filter {} to file => {}\n\t'.format(model,file)\
		 + str([str(interval) for interval in intervalsReal]),'info') 

	return file

def write_integration_problem_to_file(intervalsReal, tmpDir, nbRealInt):
	file = tmpDir + 'functions/int_func_{}_{}.py'.format(nbRealInt, len(intervalsReal))
	with open(file, 'w') as f:
		f.write('from math import inf\n\n')
		for interval in intervalsReal:
			for line in interval.get_function_string():
				f.write(line)
			f.write('\n')

	return file


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _readBoundFile(ourfile):
	nbBounds = int(ourfile.split('_')[-1].split('.')[0])
	spec = importlib.util.spec_from_file_location("bfs", ourfile)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	bounds = []
	for i in range(nbBounds):
		bounds.append(getattr(module, 'b{}'.format(i)))
	return bounds

def _readWFIle(ourfile):
	spec = importlib.util.spec_from_file_location("wf", ourfile)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module.trueWeightFunc


def compute_scipy_integration_for_intervals(weightFunctionFile, boundFile, intervalsBool,nbInt, intError,logger):
	# intervalsAsFunctions = [x.as_func(logger) for x in intervalsReal]

	start_time = time.time()
	weightFunction = _readWFIle(weightFunctionFile)
	try:
		intervalsAsFunctions = _readBoundFile(boundFile)
	except Exception as e:
		logger.error('reading bound file error: {} from file {}'.format(e, boundFile))
		raise e

	try:
		if not intervalsAsFunctions:
			(vol, error)  = (weightFunction(*intervalsBool), 0)
		else:
			(vol, error) = integrate.nquad(weightFunction, intervalsAsFunctions, args = intervalsBool, \
				opts = {'epsabs': intError[0], 'epsrel': intError[1]})
	except Exception as e:
		logger.error('integration error: {} from file: {}'.format(e, boundFile))
		raise e

	intTime = time.time() - start_time
	threadId =  mp.current_process()

	if vol == 0:
		logger.writeToLog(' == Integrate: ({},{},{:.2}, {}) x {} ==\n\t== {} || {} || {} on thread: {}'.format(vol, error,intTime, nbInt,intError,\
		weightFunction,intervalsAsFunctions, intervalsBool, threadId),'info')	
	else:
		logger.writeToLog(' == Integrate: ({:.3},{:.2},{:.2}, {}) x {} ==\n\t== {} || {} || {} on thread: {}'.format(vol, error,intTime, nbInt,intError,\
			weightFunction,intervalsAsFunctions, intervalsBool, threadId),'info')	

	vol = vol * nbInt

	# os.remove(boundFile)

	return (vol, error)




