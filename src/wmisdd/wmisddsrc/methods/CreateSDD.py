
import z3
import itertools

import platform
from subprocess import STDOUT, check_output, TimeoutExpired
import time
from ..elements.myExceptions import *
from ..manager.EnumerationManager import EnumerationManager


def convert_to_cnf(pkbAsZ3,logger):
	#t = Then('simplify','nnf')
	#subgoal = t(simplify(self._kb))
	#self._logger.writeToLog("subgoal",subgoal)
	cnf = z3.simplify(_convert_to_cnf(pkbAsZ3,logger))
	cnflist = []
	if z3.is_and(cnf):
		for i in cnf.children():
			tmp = []
			if z3.is_or(i):
				for ii in i.children():
					if z3.is_const(ii) or z3.is_not(ii) and z3.is_const(ii.children()[0]):
						tmp.append(ii)
					else:
						logger.writeToLog("Wrongly formulated CNF")
						raise Exception
			elif z3.is_not(i) and z3.is_const(i.children()[0]):
				tmp.append(i)
			elif z3.is_const(i):
				tmp.append(i)
			else:
				logger.writeToLog("Wonrgly formulated CNF")
			cnflist.append(tmp)
	elif z3.is_or(cnf):
		tmp = []
		for i in cnf.children():
			if z3.is_const(i) or z3.is_not(i) and z3.is_const(i.children()[0]):
				tmp.append(i)
			else:
				logger.writeToLog("Wonrgly formulated CNF")
		cnflist.append(tmp)
	else:
		tmp = [cnf]
		cnflist.append(tmp)

	logger.writeToLog("Full Propositional KB in CNF: {}".format(cnflist))
	return cnflist
	#self._logger.writeToLog("RESULT: CNF",cnf)
	# return subgoal[0]

def _convert_to_cnf(formula, logger):
	if z3.is_or(formula):
		tmp = []
		ground = []
		for i in formula.children():
			tmp.append(_convert_to_cnf(i,logger))
		for i in tmp:
			if z3.is_and(i):
				ground.append(i.children())
			elif z3.is_const(i):
				ground.append([i])
			elif z3.is_not(i) and z3.is_const(i.children()[0]):
				ground.append([i])
			elif z3.is_or(i) and all(z3.is_const(elem) or z3.is_not(elem) and z3.is_const(elem.children()[0]) for elem in i.children()):
				for j in i.children():
					ground.append([j])
			else:
				logger.writeToLog("is_or, {},{}".format(formula,i))
				raise Exception
		result = []
		logger.writeToLog("CROSS: {}".format(ground))
		for i in itertools.product(*ground):
			logger.writeToLog('Writing to rsults: {},{}'.format(i,list(i)))
			result.append(z3.Or(i))
		logger.writeToLog('Resutl: {}'.format(result))
		result = z3.And(result)
		logger.writeToLog('ResutAnd: {}'.format(result))
		resultS = z3.simplify(result)
		logger.writeToLog("Result simplified: {}".format(resultS))
		return resultS

	elif z3.is_and(formula):
		tmp = []
		ground = []
		for i in formula.children():
			tmp.append(_convert_to_cnf(i,logger))
		for i in tmp:
			if z3.is_and(i):
				ground.extend(i.children())
			elif z3.is_const(i):
				ground.append(i)
			elif z3.is_not(i) and z3.is_const(i.children()[0]):
				ground.append(i)
			elif z3.is_or(i) and all(z3.is_const(elem) or z3.is_not(elem) and z3.is_const(elem.children()[0]) for elem in i.children()):
				ground.append(i)

			# SHoueld be ----> (1 v 2) and 3 --> (1 and 3 or 2 and 3) not just adding them to the and statement.... right ?
			else:
				logger.error("is_and, {}, {}".format(formula, i))
				raise Exception
		return z3.simplify(z3.And(ground))
	elif z3.is_not(formula):
		if z3.is_const(formula.children()[0]):
			return formula
		elif z3.is_not(formula.children()[0]):
			return _convert_to_cnf(formula.children()[0],logger)
		elif z3.is_and(formula.children()[0]):
			return _convert_to_cnf(z3.Or([z3.Not(elem) for elem in formula.children()[0].children()]), logger)
		elif z3.is_or(formula.children()[0]):
			return _convert_to_cnf(z3.And([z3.Not(elem) for elem in formula.children()[0].children()]), logger)
		else:
			logger.writeToLog("is_not({}) problem".formula(formula))
			raise Exception
	elif z3.is_const(formula):
		return formula
	else:
		logger.writeToLog("is_nothing problem formula: {}".format(formula),'error')
		raise Exception

def save_to_demacs(file, cnfFormula, numOfPredicates):
	#self._logger.test("SAVING KB TO DEMACS: {}".format(file))
	f = open('{}'.format(file),'w')
	f.write('c this the Abstacted SMT Formalu in CNF (DEMACS)\n')
	f.write('p cnf {} {}\n'.format(numOfPredicates, len(cnfFormula)))
	for disj in cnfFormula:
		for var in disj:
			if z3.is_not(var):
				elem = var.children()[0]
				pre = '-'
			else:
				elem = var
				pre = ''
			f.write('{}{} '.format(pre,str(elem)))
		f.write('0\n')
	f.close()

def parse_sdd(BIN_DIR, cnfFile,sddFile, vtreeFile, timeout, logger, precomputed_vtree = False):
	vtreeFlag = '-v' if precomputed_vtree else '-W'
	vtreeSearchFlag = '-r 0' if precomputed_vtree else ''
	vtreeDotFile = vtreeFile + '.dot'
	sddDotFiel = sddFile + '.dot'
	if 'Linux' in platform.system():
		command = "{}/sdd-linux -c {} {} {} -R {} {} -m -V {} -S {}".format(BIN_DIR,cnfFile,\
			vtreeFlag, vtreeFile, sddFile, vtreeSearchFlag, vtreeDotFile, sddDotFiel)
	else:
		command = "{}/sdd-darwin -c {} {} {} -R {} {} -m -V {} -S {}".format(BIN_DIR, cnfFile,\
			vtreeFlag, vtreeFile, sddFile,vtreeSearchFlag, vtreeDotFile, sddDotFiel)
	#logging.debug('\t' + command)

	start_time = time.time()
	try:
		print('\nexecuting command: {}'.format(command))
		output = check_output(command, stderr=STDOUT, timeout=timeout,shell = True)
		print('finished with output: {}\n'.format(output.strip()))
		end_time = time.time()
		compileTime = end_time - start_time
	except TimeoutExpired as e:
		end_time = time.time()
		compileTime = end_time - start_time
		logger.error("Compiling into SDD is terminated due to to a timeout [Problem: {}]".format(cnfFile))
		raise TimeoutException
	except Exception as e:
		end_time = time.time()
		compileTime = end_time - start_time
		logger.error("Some Unknown Error occured for problem: {}, \n\t Ex:{}".format(cnfFile, e))
		raise e
	return compileTime

def read_sdd_from_file(name,vtreeFile, sddFile, tmpDir, logger):
	return EnumerationManager(name,vtreeFile, sddFile, tmpDir, logger)