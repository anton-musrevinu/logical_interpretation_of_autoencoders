from ..elements.Interval import Interval
import itertools

def create_real_interval(leadVar, model,referencedPredicats, logger):

	interval = Interval(leadVar)

	trueRefPredicates = []
	falseRefPredicates = []
	
	for predicate in referencedPredicats:
		assignment = model[predicate.get_id() - 1] == 1
		if assignment:
			trueRefPredicates.append(predicate.get_id())
		else:
			falseRefPredicates.append(predicate.get_id())

		interval.combine_bound(predicate.get_bound(), assignment)
		if interval.is_zero():
			logger.writeToLog('Interval for modelForRealVars:' + str(model) + ',\tleadVar: {}\t, interval: {}, ref T: {}, F: {} --> Skipping Model'.format(\
					leadVar, str(interval), trueRefPredicates, falseRefPredicates),'debug')
			return False

	logger.writeToLog('Interval for modelForRealVars: ' + str(model) + ',\tleadVar: {}\t, interval: {}, ref T: {}, F: {}'.format(\
					leadVar, str(interval), trueRefPredicates, falseRefPredicates),'info')

	return interval

def get_Bool_Interval(boolVar,model, predicatesToIndx):
	if boolVar in predicatesToIndx:
		truth = model[predicatesToIndx[boolVar] -1] == 1
		interval = [truth]
	else:
		interval = [True, False]
	return interval

def complete_bool_intervals(intervalsBool):
	#numIntervals = sum([len(x) for x in intervalsBool])
	fullTable = itertools.product([False, True], repeat=len(intervalsBool))
	actualBoolIntervlas = []
	for row in fullTable:
		skip = False
		for i,elem in enumerate(row):
			if not elem in intervalsBool[i]:
				skip = True
				break
		if skip == True:
			continue
		actualBoolIntervlas.append(tuple(row))

	return actualBoolIntervlas