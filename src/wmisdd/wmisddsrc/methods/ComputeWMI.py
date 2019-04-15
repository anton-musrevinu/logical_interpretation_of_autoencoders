
from collections import OrderedDict
from ..elements.Interval import Interval
from .LatteIntegration import create_latte_file_for_model,init_latte_weight_function,compute_latte_integration_for_intervals, create_latte_file_for_model_new
from .ScipyIntegration import create_scipy_file_for_model,init_scipy_weight_function,compute_scipy_integration_for_intervals
from .IntegrationEssentials import complete_bool_intervals, get_Bool_Interval
from ..elements.myExceptions import IntegrationError
import itertools
import time
from scipy import integrate
import os
import multiprocessing as mp
import importlib


DEF_COMUTE_SCIPY = 'DEF_COMUTE_SCIPY'
DEF_COMUTE_LATTE = 'DEF_COMUTE_LATTE'

def _del_tmp_func_files(tmpFuncDir,model):
	f = []
	for (dirpath, dirnames, filenames) in os.walk(tmpFuncDir):
		f.extend(filenames)
		break

	for file in f:
		if file.startswith('m{}'.format(model)):
			os.remove(tmpFuncDir + file)

def construct_integrateion_data(models, integrationMethod, hkb, wf, tmpDir, logger):

	if integrationMethod == DEF_COMUTE_SCIPY:
		os.mkdir(tmpDir + 'functions/')

	numVar = hkb.get_num_of_predicates()
	realsVarOrder = wf.get_real_variable_order()
	boolsVarOrder = wf.get_bool_variable_order()

	intervalsData = {}
	intervalsDataReals = {}
	for model in models:
		model = model[:numVar]
		modelForRealVars = (model & hkb.get_continuous_predicate_filter())
		# print(str(model),self._hkb.get_continuous_predicate_filter(),str(modelForRealVars))

		if str(modelForRealVars) in intervalsDataReals:
			intervalsRealFile = intervalsDataReals[str(modelForRealVars)]
		else:
			intervalsRealFile = create_real_intervals_for_model(modelForRealVars, hkb, realsVarOrder,\
				 boolsVarOrder, tmpDir, integrationMethod, logger)
			intervalsDataReals[str(modelForRealVars)] = intervalsRealFile

		if intervalsRealFile == False:
			continue

		intervalsBoolList = create_bool_intervals_for_model(model, hkb, boolsVarOrder, logger)

		for intervalsBool in intervalsBoolList:
			intervalsIdentifier = (str(modelForRealVars),intervalsBool)
			if intervalsIdentifier in intervalsData:
				intervalsData[intervalsIdentifier][2] += 1
			else:
				intervalsData[intervalsIdentifier] = [intervalsRealFile, intervalsBool,1]
			logger.writeToLog('Updating Integration Instances: {}'.format(intervalsData[intervalsIdentifier]),'info')

	return intervalsData.values(), len(intervalsData.values())

def create_real_intervals_for_model(model, hybridKnowlegeBase, orderedReals,orderedBools,tmpDir,integrationMethod, logger):
	if integrationMethod == DEF_COMUTE_SCIPY:
		return create_scipy_file_for_model(model, hybridKnowlegeBase, orderedReals,orderedBools,tmpDir, logger)
	elif integrationMethod == DEF_COMUTE_LATTE:
		return create_latte_file_for_model_new(model, hybridKnowlegeBase, orderedReals, tmpDir, logger)
	else:
		raise IntegrationError('Unknown integrationMethod: {}'.format(integrationMethod))

def create_wf_file(weightFunction, tmpDir, integrationMethod, logger):
	if weightFunction.get_function_as_z3() == None:
		raise IntegrationError('Trying to initialize WF, but WF has not been abstracted yet')
	if weightFunction.get_real_variable_order() == None:
		raise IntegrationError('Trying to initialize WF, but Real variable Order has not been constructed yet')
	if weightFunction.get_bool_variable_order() == None:
		raise IntegrationError('Trying to initialize WF, but Real variable Order has not been constructed yet')

	if integrationMethod == DEF_COMUTE_SCIPY:
		return init_scipy_weight_function(weightFunction, tmpDir, logger)
	elif integrationMethod == DEF_COMUTE_LATTE:
		return init_latte_weight_function(weightFunction, tmpDir, logger)
	else:
		raise IntegrationError('Unknown integrationMethod: {}'.format(integrationMethod))

def create_bool_intervals_for_model(model, hybridKnowlegeBase, orderedBools, logger):

	intervalsBool = []
	for var in orderedBools:
		interval = get_Bool_Interval(var, model, hybridKnowlegeBase.get_bool_predicate_to_indx())
		intervalsBool.append(interval)

	intervalsBoolList = complete_bool_intervals(intervalsBool)


	logger.writeToLog('Created intervalsBoolList for model {} => \n\t'.format(model)\
		 + str(intervalsBoolList),'debug')

	return intervalsBoolList

#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------
#------------------------------------------------------  Integration -------------------------------------------------------------------

# def compute_vol(weightFunction, intervalsReal, intervalsBoolList, computedIntervals, intError, logger):

# 	start_time = time.time()

# 	totalVol = 0
# 	totalErr = 0

# 	tmplen = len(computedIntervals)

# 	if intervalsReal == [] and intervalsBoolList == [()]:
# 		totalVol = weightFunction.get()()
# 		totalErr = 0
# 	else:
# 		for intervalsBool in intervalsBoolList:

# 			intervalIdentifier = construct_interval_identifier(intervalsReal, intervalsBool)
# 			if intervalIdentifier in computedIntervals.keys():
# 				(vol, error) = computedIntervals[intervalIdentifier]
# 			else:
# 				(vol, error) = compute_integration_for_intervals(weightFunction,intervalsReal, intervalsBool, intError,logger)
# 				computedIntervals[intervalIdentifier] = (vol, error)

# 			totalVol += vol
# 			totalErr += error

# 	end_time = time.time()

# 	return (totalVol, totalErr, ((end_time - start_time), len(computedIntervals) - tmplen))

# def construct_interval_identifier(intervalsReal, intervalsBool):

# 	intervalsAsString = [str(x) for x in intervalsReal]
# 	intervalIdentifier = str(intervalsAsString) + "-" + str(intervalsBool)

# 	return intervalIdentifier


def compute_integration_for_intervals(weightFunctionFile, boundFile, intervalsBool,nbInt, intError,\
	integrationMethod,integrationProblemID,displayProgressIds,tmpDir, logger):
	try:
		if integrationMethod == DEF_COMUTE_SCIPY:
			result =  compute_scipy_integration_for_intervals(\
				weightFunctionFile, boundFile, intervalsBool,nbInt, intError, logger)
		elif integrationMethod == DEF_COMUTE_LATTE:
			result = compute_latte_integration_for_intervals(\
				weightFunctionFile, boundFile, intervalsBool,nbInt,integrationProblemID, tmpDir, logger)
		else:
			raise Exception('Unknown integrationMethod: {}'.format(integrationMethod))

		if integrationProblemID in displayProgressIds:
			logger.writeToLog('\t\t\t Integration Progression: {}%'.format(\
				displayProgressIds.index(integrationProblemID)/len(displayProgressIds) * 100),'result')

		return result
	except Exception as e:
		logger.error(str(e))
