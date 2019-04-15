from ..methods import AbstractionKB
from ..methods import AbstractionWF
from ..methods import RewritePredicates
from ..methods import CreateSDD
from ..methods import QuerySDD
from ..methods import ComputeWMI

from ..elements import myExceptions

from ..elements.Results import WmiResult
from ..elements.Interval import Interval

import time

from joblib import Parallel, delayed
import importlib
#from multiprocessing import Pool
import multiprocessing as mp
import os
from shutil import rmtree
import itertools
import z3
import numpy as np

class WMIManager(object):

	DEF_TIMEOUT_SDDCOMPILE = 2 * 60 * 60
	DEF_TIMEOUT_ME = 2 * 60 * 60
	DEF_TIMOOUT_INT = 3 * 60 * 60

	def __init__(self,name,tmpDir,logger):
		# this dictonary identifies a number with a propositional letter and the abstraction
		self._logger = logger
		self._name = name
		self._tmpDir = tmpDir

		self._sdd = None
		self._wf = None
		self._hkb = None
		self._models = None
		self._computedIntervals = {}
		self._result = WmiResult(name)

		self.BIN_DIR = os.path.abspath('../src/wmisdd/bin')

		if os.path.isdir(self._tmpDir):
			rmtree(self._tmpDir)
		os.mkdir(self._tmpDir)

	def get_hkb(self):
		return self._hkb

	def get_models(self):
		return self._models

	def abstract(self, hkbAsZ3, wfAsZ3 = z3.RealVal(1), keep_original_names = False):
		try:
			start_time = time.time()

			self._hkb = AbstractionKB.init_hkb(self._name, keep_original_names)
			AbstractionKB.abstract_hkb(hkbAsZ3,self._logger, self._hkb)
			self._hkb.store_tmp_dir(self._tmpDir)

			self._wf = AbstractionWF.abstract_single_wf(wfAsZ3, self._logger)

			end_time = time.time()
			execTime = end_time - start_time
			self._result.set_abstraction_time(execTime)

			self._logger.writeToLog('\t- finished abstraction [{}]'.format(execTime),'benchmark')

			return self._result

		except Exception as e:
			self._logger.error('Abstraction of the KB and w the following error: {}'.format(e))
			raise myExceptions.AbstractionError

	def abstract_query_and_add(self,queryString):

		start_time = time.time()

		try:
			self._hkb = AbstractionKB.abstract_hkb(queryString, self._logger, self._hkb)

			end_time = time.time()
			execTime = end_time - start_time
			self._result.set_abstraction_time(execTime)

			return self._result

		except Exception as e:
			self._logger.error('Abstraction of the query produced the following error: {}'.format(e))
			raise myExceptions.AbstractionError

	def abstract_and_abstract_query(self, hkbString, wString, queryString):

		start_time = time.time()

		self.abstract(hkbString, wString)

		try:
			self._hkb = AbstractionKB.abstract_hkb(queryString, self._logger, self._hkb)

			end_time = time.time()
			execTime = end_time - start_time
			self._result.set_abstraction_time(execTime)

			return self._result

		except Exception as e:
			self._logger.error('Abstraction of the query produced the following error: {}'.format(e))
			raise myExceptions.AbstractionError

	def rewrite_atoms(self):

		try:

			start_time = time.time()

			orderedRealVariables, orderedBoolVariables = RewritePredicates.construct_variable_order(self._wf, self._wf.DEF_VAR_ORDER_ALPH)
			self._wf.set_variable_order(orderedRealVariables, orderedBoolVariables)

			RewritePredicates.rewrite_all_predicates(self._hkb, orderedRealVariables, self._logger)
			self._hkb.update_leadVar_dict()

			execTime = time.time() - start_time

			self._result.set_rewrite_time(execTime)

			self._logger.writeToLog('\t- finished rewriting [{}]'.format(execTime),'benchmark')

		except Exception as e:
			self._logger.error('Rewriting the proposititional refinements produced the following error: {}'.format(e))
			raise myExceptions.RewritingError

	def find_conditions(self):
		try:

			start_time = time.time()
			conditions = []

			#combinations: every variable with every other variable - var**4 - Pos and neg
			realsVarOrder = self._wf.get_real_variable_order()
			for leadVar in realsVarOrder:
				referencedPredicates = list(self._hkb.get_referenced_predicates(leadVar))
				referencedPredicatesCopy = referencedPredicates[:]
				seen = []
				for (p1,p2) in itertools.product(referencedPredicates, referencedPredicatesCopy):
					if p1 == p2 or (p1,p2) in seen or (p2,p1) in seen:
						continue
					seen.append((p1,p2))
					for (p1assign, p2assign) in [(True,True),(False, True), (True, False), (False, False)]:
						interval = Interval(leadVar)
						interval.combine_bound(p1.get_bound(), p1assign)
						interval.combine_bound(p2.get_bound(), p2assign)
						if interval.is_zero():
							c1 = p1.getBoolRef() if p1assign else z3.Not(p1.getBoolRef())
							c2 = p2.getBoolRef() if p2assign else z3.Not(p2.getBoolRef())
							condition = z3.simplify(z3.Or(z3.Not(c1),z3.Not(c2)))
							# self._logger.writeToLog('adding condition: p1: ({},{}), p2: ({},{}), {}'.format(p1,p1assign,p2,p2assign,condition),'test')
							conditions.append(condition)

			# self._logger.writeToLog('appending the conditions to the pkb:','test')
			# self._logger.writeToLog('\tpkb: {}'.format(self._hkb.get_prop_kb()),'test')
			# self._logger.writeToLog('\tconditions: {}'.format(z3.And(conditions)),'test')
			if conditions:
				self._hkb.append_pkb_as_z3(z3.And(conditions))
			self._result.set_condition_time(time.time() - start_time)
			# self._logger.writeToLog('\ttogether: {}'.format(self._hkb.get_prop_kb()),'test')

		except Exception as e:
			self._logger.error('Finding additional conditions produced the following error: {}'.format(e))
			raise e


	def create_sdd(self, timeout = DEF_TIMEOUT_SDDCOMPILE, sdd_file = None, vtree_file = None, total_num_vars = None,precomputed_vtree = False, cnf_dir = None):

		start_time = time.time()
		try:
			#retrieving the propositional KB
			pkb = self._hkb.get_prop_kb()
			self._logger.writeToLog('Creating CNF with the KB: {}'.format(self._hkb), 'info')

			#Converting the propositional KB in cnf format (list of Conjunvtions)
			cnf = CreateSDD.convert_to_cnf(pkb,self._logger)

			self._logger.writeToLog('Creating SDD with CNF: {}'.format(cnf),'info')

			#Saving the cnf to a .cnf file in the tmp directory
			if cnf_dir != None:
				cnfTmpFile = os.path.join(cnf_dir, 'constrains.cnf')
			else:
				cnfTmpFile = os.path.join(self._tmpDir,'cnf_tmp_file.cnf')

			if total_num_vars != None:
				numOfPredicates = total_num_vars
			else:
				numOfPredicates = self._hkb.get_num_of_predicates()
			CreateSDD.save_to_demacs(cnfTmpFile, cnf, numOfPredicates)

			#Compiling the cnf file into a sdd and vtree file
			if sdd_file == None:
				sdd_file = os.path.join(self._tmpDir,'sdd_tmp_file.sdd')
			if vtree_file == None:
				vtree_file = os.path.join(self._tmpDir, 'vtree_tmp_file.vtree')
			compileTime = CreateSDD.parse_sdd(self.BIN_DIR, cnfTmpFile,sdd_file, vtree_file,timeout,self._logger,precomputed_vtree = precomputed_vtree)
			self._logger.writeToLog('SDD was created at: {}'.format(sdd_file),'result')
			if not precomputed_vtree:
				self._logger.writeToLog('vtree was created at: {}'.format(vtree_file),'result')

			#Creating the SDD file structure (possible addition, reading the sdd into the internal file format)
			self._sdd = CreateSDD.read_sdd_from_file(self._name, vtree_file, sdd_file,self._tmpDir, self._logger)

			execTime = time.time() - start_time
			self._result.set_sdd_creation_time(execTime)


			self._logger.writeToLog('\t- finished KC [{}]'.format(execTime),'benchmark')

			return self._result

		except myExceptions.TimeoutException as e:
			self._result.set_indicator(WmiResult.INDICATOR_KC_TIMEOUT)
			self._result.set_sdd_creation_time(time.time() - start_time)
			return

		except Exception as e:
			self._result.set_sdd_creation_time(time.time() - start_time)
			self._result.set_indicator(WmiResult.INDICATOR_KC_UNKOWN)
			raise myExceptions.KnowledgeCompilationError

	def query_sdd(self,timeout = DEF_TIMEOUT_ME, sddTmpFile = None, vtreeTmpFile = None, setModelCount = False):
		if not self._result.is_all_good():
			return

		if self._sdd == None:
			self._sdd = CreateSDD.read_sdd_from_file(self._name, vtreeTmpFile, sddTmpFile, self._tmpDir, self._logger)


		start_time = time.time()
		try:

			#Reading the compiled sdd file (.sdd) back into the internal file struction wmisdd.elements.sdd
			#And computing all satisfying models
			self._models, execTime = QuerySDD.retriev_all_satisfying_models(self._sdd, timeout)

			# self._logger.writeToLog(str([str(a) for a in self._models]),'test')

			self._result.set_sdd_query_time(time.time() - start_time)
			if setModelCount:
				self._result.set_model_count(len(self._models))

			self._logger.writeToLog('\t- finished ME [{}]'.format(time.time() - start_time),'benchmark')

			return self._result

		except myExceptions.TimeoutException as e:
			self._result.set_indicator(WmiResult.INDICATOR_ME_TIMEOUT)
			self._result.set_sdd_query_time(time.time() - start_time)
			return

		except myExceptions.OverFlowException as e:
			self._result.set_indicator(WmiResult.INDICATOR_ME_OVERFLOW)
			self._result.set_sdd_query_time(time.time() - start_time)
			return

		except Exception as e:
			self._logger.error('query_sdd of the KB produceded following error: {}'.format(e))
			self._result.set_indicator(WmiResult.INDICATOR_UNKNOWN)
			self._result.set_sdd_query_time(time.time() - start_time)
			raise myExceptions.ModelEnumerationError

	def _parse_write(self,functionAsPythonString, tmpFile):
		with open(tmpFile,'w') as f:
			for line in functionAsPythonString:
				f.write(line)

	def compute_wmi_single_weight_par(self, intError = (10,10), integrationMethod = ComputeWMI.DEF_COMUTE_SCIPY):

		if not self._result.is_all_good():
			return

		try:
			start_time = time.time()

			wfFile = ComputeWMI.create_wf_file(self._wf, self._tmpDir, integrationMethod ,self._logger)

			if self._hkb.is_boolean() and self._wf.is_one():
				wmi = len(self._models)
			else:

				integrationData, totalNBofIntegrations = ComputeWMI.construct_integrateion_data(\
					self._models, integrationMethod, self._hkb, self._wf, self._tmpDir, self._logger)
				# filenames = ComputeWMI.wrtie_integration_problems_to_file(integrationData, self._tmpDir, self._logger)
				self._result.set_construction_time(time.time() - start_time)
				self._logger.writeToLog('\t- finished IC [{}]'.format(time.time() - start_time),'result')
				start_time = time.time()


				self._results = {}
				self._wmi = 0
				self._error = 0
				integrationProblemID = 0

				self._logger.writeToLog('Distributing the Integration Problems to the workers, currently alive are: {}\n\tProblems: {}'.format(mp.active_children(), str(integrationData)), 'info')

				num_cores = mp.cpu_count()
				self.pool =  mp.Pool(num_cores)
				self._logger.writeToLog('\t+ starting integration pool with {} slots for {} integrations:'.format(\
					num_cores, totalNBofIntegrations),'result')

				if totalNBofIntegrations >= 10:
					displayProgressIds = list(map(lambda x: int(x), np.linspace(0,totalNBofIntegrations,10)))
				else:
					displayProgressIds = list(range(totalNBofIntegrations))

				for integrationProblem in integrationData:
					[boundfile, intervalsBool, nbInt] = integrationProblem
					# print('starting worker on: {}'.format(integrationProblem))
					self.pool.apply_async(ComputeWMI.compute_integration_for_intervals, \
						args = (wfFile, boundfile, intervalsBool,nbInt, intError,\
							integrationMethod,integrationProblemID, displayProgressIds,self._tmpDir ,self._logger, )\
						, callback = self.log_resuls)
					integrationProblemID += 1
				# print('waiting for workers with: {}'.format(mp.active_children()))
				self.pool.close()
				self.pool.join()	
				self._logger.writeToLog('\t- finished integration [{}]'.format(time.time() - start_time),'result')
				self._logger.writeToLog('joined and closed the pool, currently alive workers are: {}'.format(mp.active_children()),'info')
				# print('pool apperently closed with: {} results: {}'.format(mp.active_children(),self._results))


			end_time = time.time()
			total_time = (end_time - start_time)

			self._result.set_wmi_result(self._wmi, total_time, self._error, integrationProblemID)

			#self._computedIntervals.update(abstractionManager.getComputedIntervals())
			self._logger.writeToLog('The WMI for the KB is: {}, {}'.format(self._wmi,self._result.get_times_string()),'info')
			return self._result

		except myExceptions.TimeoutException as e:
			self._result.set_indicator(WmiResult.INDICATOR_INT_TIMEOUT)
			self._result.set_wmi_int_time(time.time() - start_time)
			return

		except Exception as e:
			self._logger.error('Computing the wmi produced following error: {}'.format(e))
			raise myExceptions.AbstractionError

	def del_all_tmp_files(self):
		rmtree(self._tmpDir)

	def log_resuls(self, result):
		# print('recieved result: {}'.format(result[0]))
		if result == None:
			self.pool.terminate()
			self._wmi = None
			return
		self._wmi += result[0]
		self._error += result[1]

	def get_result(self):
		return self._result
