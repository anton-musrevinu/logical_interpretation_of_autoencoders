from ..elements.Results import QueryResult
from .WMIManager import WMIManager
from ..methods.ComputeWMI import DEF_COMUTE_LATTE
from ..parsers.ParseInputs import parse_sm1_to_z3
import copy
import time

class QueryManager:

	def __init__(self, propName, tmpDir,intError, logger):
		if propName == None:
			propName = 'QueryManager'
		self._propName = propName
		self._tmpDir   = tmpDir
		self._logger   = logger
		self._result = QueryResult(propName)
		if intError == None:
			self._intError = (3,3)
		else:
			self._intError = intError

		self._baseManager = None

	def do_wmi_base_parse(self, hkbAsString, wfAsString, integrationMethod = DEF_COMUTE_LATTE, keepFiles = False):

		hkbAsZ3 = parse_sm1_to_z3(hkbAsString, self._logger)
		wfAsZ3 = parse_sm1_to_z3(wfAsString, self._logger)

		self._logger.writeToLog('\tFinished parsing','result')

		testTime = time.time()

		self.do_wmi_base(hkbAsZ3, wfAsZ3, integrationMethod, keepFiles)

		return time.time() - testTime

	def convert_to_sdd(self, hkbAsZ3, sdd_file = None, vtree_file = None, printModels = False, total_num_vars = None, precomputed_vtree = False, cnf_dir = None):
		# print(hkbAsZ3)
		start_time = time.time()
		wmiManager = WMIManager(self._propName, self._tmpDir, self._logger)
		wmiManager.abstract(hkbAsZ3, keep_original_names = True)
		wmiManager.rewrite_atoms()
		wmiManager.find_conditions()

		wmiManager.create_sdd(sdd_file = sdd_file ,vtree_file = vtree_file, total_num_vars = total_num_vars, precomputed_vtree = precomputed_vtree, cnf_dir = cnf_dir)
		total_time = time.time() - start_time
		if printModels:
			wmiManager.query_sdd()
			for model in wmiManager.get_models():
				print('\t' + str(model[:total_num_vars]))

	def do_enumeration(self,vtree_file, sdd_file, total_num_vars):
		wmiManager = WMIManager(self._propName, self._tmpDir, self._logger)
		wmiManager.read_from_file(vtree_file, sdd_file)
		wmiManager.query_sdd(setModelCount = True)
		print('model count: {}'.format(wmiManager.get_model_count()))
		for model in wmiManager.get_models():
			print('\t' + str(model[:total_num_vars]))

	def do_wmi_base(self, hkbAsZ3, wfAsZ3, integrationMethod = DEF_COMUTE_LATTE, keepFiles = False):

		start_time = time.time()

		wmiManager = WMIManager(self._propName, self._tmpDir, self._logger)
		wmiManager.abstract(hkbAsZ3, wfAsZ3)
		wmiManager.rewrite_atoms()
		wmiManager.find_conditions()
		wmiManager.create_sdd()
		wmiManager.query_sdd()
		wmiManager.compute_wmi_single_weight_par(self._intError, integrationMethod = integrationMethod)
		if not keepFiles:
			wmiManager.del_all_tmp_files()

		total_time = time.time() - start_time
		result = wmiManager.get_result()
		result.set_total_time(total_time)
		
		self._result.add_base_wmi_result(result)
		self._baseManager = wmiManager

	def do_wmi_query(self,queryString):
		return self.do_wmi_query_new(queryString)

	def do_wmi_query_new(self, queryString):
		propName = self._propName + '-Query'

		wmiManager =  WMIManager(propName, self._tmpDir, self._logger)
		wmiManager.abstract_and_abstract_query(self._baseHkbString, self._baseWfString, queryString)
		wmiManager.rewrite_atoms()
		wmiManager.create_sdd()
		wmiManager.query_sdd()
		wmiManager.compute_wmi_single_weight_par(self._intError, integrationMethod = integrationMethod)

		self._result.add_query_wmi_result(wmiManager.get_result())

		self._result.check_resulting_prob()

		return self._result

	def get_result(self):
		return self._result
