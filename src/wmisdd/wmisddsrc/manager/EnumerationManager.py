
#Manager
from ..sdd.Algorithm_ModelCounting_Basic import Algorithm_ModelCounting_Basic
from ..sdd.Algorithm_ModelCounting_Par import Algorithm_ModelCounting_Par
from ..sdd.Algorithm_ModelEnumeration_Ram_List import Algorithm_ModelEnumeration_Ram
from ..sdd.Algorithm_ModelEnumeration_Disk import Algorithm_ModelEnumeration_Disk
from ..sdd.Algorithm_ModelEnumeration_Pure import Algorithm_ModelEnumeration_Pure
from ..sdd.Algorithm_ModelEnumeration_Par_FULL import Algorithm_ModelEnumeration_Par
from ..sdd.SddBaseManager import SddBaseManager
from ..elements.myExceptions import StopException, AlgorithmInitException, TimeoutException, OverFlowException

from ..elements.mylogger import Mylogger
import psutil
from multiprocessing import Process, Pipe
import multiprocessing as mp
from subprocess import STDOUT, check_output, TimeoutExpired
import time
import gc
import threading
import traceback
import logging
import platform
import os.path
import importlib

class EnumerationManager:
	"""The manager, used ot create, read, query and handle Sdds and Vtrees internally.
	"""

	DEF_TIMEOUT_MC = 1 * 60
	DEF_TIMEOUT_ME = 10 * 60
	DEF_TIMEOUT_SDDCOMPILE = 2 * 60

	MEMORY_THRESHOLD = 90

	INDICATOR_OVERFLOW = -1
	INDICATOR_TIMEOUT = -2
	INDICATOR_ALLGOOD = 0
	INDICATOR_UNKNOWN = -3

	ALGORITHM_MC_BASIC = 'ALGORITHM_MC_BASIC'
	ALGORITHM_MC_PAR = 'ALGORITHM_MC_PARALLEL'
	ALGORITHM_ME_RAM = 'ALGORITHM_ME_RAM'
	ALGORITHM_ME_DISK = 'ALGORITHM_ME_DISK'
	ALGORITHM_ME_PURE = 'ALGORITHM_ME_PURE'
	ALGORITHM_ME_PAR_DISK = 'ALGORITHM_ME_PAR_DISK'
	ALGORITHM_ME_PAR_RAM = 'ALGORITHM_ME_PAR_RAM'


	def __init__(self,name,vtreeTmpFile,sddTmpFile,tmpDir, logger):
		self._sddManager = None
		self._name = name
		self._vtreeFile = vtreeTmpFile
		self._sddFile= sddTmpFile
		self._tmpDir = tmpDir
		self._logger = logger

	def initAlgorithm(self,algorithm = ALGORITHM_MC_BASIC, verbose = False,threads = 1):

		if algorithm == EnumerationManager.ALGORITHM_MC_BASIC:
			self._sddManager = Algorithm_ModelCounting_Basic(self._name, self._logger)
		elif algorithm == EnumerationManager.ALGORITHM_MC_PAR:
			self._sddManager = Algorithm_ModelCounting_Par(self._name, self._logger,threads = threads)
		elif algorithm == EnumerationManager.ALGORITHM_ME_RAM:
			self._sddManager = Algorithm_ModelEnumeration_Ram(self._name, self._logger)
		elif algorithm == EnumerationManager.ALGORITHM_ME_DISK and self._tmpDir != None:
			self._sddManager = Algorithm_ModelEnumeration_Disk(self._name, self._logger,self._tmpDir)
		elif algorithm == EnumerationManager.ALGORITHM_ME_PURE:
			self._sddManager = Algorithm_ModelEnumeration_Pure(self._name, self._logger)
		elif algorithm == EnumerationManager.ALGORITHM_ME_PAR_RAM:
			num_cores = mp.cpu_count()
			self._sddManager = Algorithm_ModelEnumeration_Par(self._name, self._logger,\
				threads = num_cores, version = Algorithm_ModelEnumeration_Par.VERSION_RAM)
		elif algorithm == EnumerationManager.ALGORITHM_ME_PAR_DISK and self._tmpDir != None:
			num_cores = mp.cpu_count()
			self._sddManager = Algorithm_ModelEnumeration_Par(self._name, self._logger,\
				pathToModels = self._tmpDir,threads = num_cores, version = Algorithm_ModelEnumeration_Par.VERSION_DISK)
		else:
			raise Exception("Algorithm \"{}\" not known.[type: {}]".format(algorithm,type(algorithm)))

		self._logger.setVerbose(verbose)
		self._sddManager.initTree(self._vtreeFile, self._sddFile)

	def checkIfFilesExist(self, fileType):
		'''Simple funciton to check if the file we are about to use exists
			works for sdd/vtree and cnf fiels
			args: fileType - > is a string, either sdd or cnf inidcating what file we are checking
			returns: true ro false corresponding to the file existing or not'''

		if fileType == "sdd":
			if not os.path.isfile(self._vtreeFile):
				return self._vtreeFile
			self._logger.writeToLog('Vtree File found: {}'.format(self._vtreeFile))
			if not os.path.isfile(self._sddFile):
				return self._sddFile
			self._logger.writeToLog('SDD File found: {}'.format(self._sddFile))
		elif fileType == 'cnf':
			if not os.path.isfile(self._cnfFile):
				return self._cnfFile
			self._logger.writeToLog('CNF File found: {}'.format(self._cnfFile))
		return True


	'''-----------------------------------------------------------------------------------------'''
	'''----------                            Vtree Methods                            ----------'''
	'''-----------------------------------------------------------------------------------------'''

	def getTotalVariabelCount(self):
		if self._sddManager:
			return self._sddManager.getVarCount()

	def _compileToSdd(self, timeout):
		if 'Linux' in platform.system():
			command = "./../c/bin/sdd-linux -c {0} -W {1} -R {2} -r 5".format(self._cnfFile,\
				self._vtreeFile, self._sddFile)
		else:
			command = "./../c/bin/sdd-darwin -c {0} -W {1} -R {2} -r 5".format(self._cnfFile,\
				self._vtreeFile, self._sddFile)
		#logging.debug('\t' + command)

		start_time = time.time()
		try:
			output = check_output(command, stderr=STDOUT, timeout=timeout,shell = True)
			end_time = time.time()
			compileTime = end_time - start_time
		except TimeoutExpired as e:
			end_time = time.time()
			compileTime = end_time - start_time
			self._logger.writeToLog("Compiling into SDD is terminated due to to a timeout [Problem: {}]".format(self._cnfFile))
			return EnumerationManager.INDICATOR_TIMEOUT,compileTime
		except Exception as e:
			end_time = time.time()
			compileTime = end_time - start_time
			self._logger.error("Some Unknown Error occured for problem: {}, \n\t Ex:{}".format(self._cnfFile, e))
			return EnumerationManager.INDICATOR_UNKNOWN, compileTime
		return EnumerationManager.INDICATOR_ALLGOOD,compileTime

	'''-----------------------------------------------------------------------------------------'''
	'''----------                      SDD Methods                                    ----------'''
	'''-----------------------------------------------------------------------------------------'''

	def _readSdd(self,file, verbose = False):
		if not self._vtreeManager:
			return 'Vtree manager not initialized, please load the vtree first.'
		self._logger.setVerbose(verbose)
		self._sddManager = SddManager(self._name,self._vtreeManager, self._logger)
		self._sddManager.fromFile(file)

	def getDepth_Sdd(self):
		if self._sddManager:
			return self._vtreeManager.getDepth()

	def getNodeCount(self):
		if self._sddManager:
			return self._sddManager.getNodeCount()

	def getBoundVariableCount(self):
		if self._sddManager:
			return self._sddManager.getVarCount()

	def getSddSize(self):
		if self._sddManager:
			return self._sddManager.getSize()


	'''-----------------------------------------------------------------------------------------'''
	'''----------                      Query SDD Methods                              ----------'''
	'''-----------------------------------------------------------------------------------------'''

	def compileSDD(self,timeout = DEF_TIMEOUT_SDDCOMPILE, verbose = False):
		return self._compileToSdd(timeout)

	def getModelCount(self,algorithm = ALGORITHM_MC_BASIC, timeout = DEF_TIMEOUT_MC, verbose = False, threads = 0):

		if not (algorithm == EnumerationManager.ALGORITHM_ME_RAM or \
				algorithm == EnumerationManager.ALGORITHM_MC_BASIC or \
				algorithm == EnumerationManager.ALGORITHM_ME_PURE or \
				algorithm == EnumerationManager.ALGORITHM_ME_DISK ):
			raise Exception("Algorithm \"{}\" not supported for Model Counting.[type: {}]".format(algorithm,type(algorithm)))

		if self._sddManager and self._sddManager.isResultComputed():
			return self._sddManager.getModelCount()
		elif self._sddManager == None:
			self.initAlgorithm(algorithm = algorithm, verbose = verbose, threads = threads)

		indicator,execTime = self._enumerateAlgorithm(algorithm, timeout,verbose, threads)

		if indicator == EnumerationManager.INDICATOR_ALLGOOD:
			return self._sddManager.getModelCount(), execTime
		else:
			return indicator, execTime

	def getModelEnumerate(self,algorithm = ALGORITHM_ME_PAR_DISK, timeout = DEF_TIMEOUT_ME, verbose = False, threads = 0):
		if not (algorithm == EnumerationManager.ALGORITHM_ME_DISK or \
				algorithm == EnumerationManager.ALGORITHM_ME_RAM or \
				algorithm == EnumerationManager.ALGORITHM_ME_PURE or \
				algorithm == EnumerationManager.ALGORITHM_ME_PAR_DISK):
			raise Exception("Algorithm \"{}\" not supported for Model Enumeration.[type: {}]".format(algorithm,type(algorithm)))

		if self._sddManager and self._sddManager.isResultComputed():
			return (Manager.INDICATOR_ALLGOOD, self._sddManager.getModels(), 0)
		elif self._sddManager == None:
			self.initAlgorithm(algorithm = algorithm, verbose = verbose, threads = threads)


		indicator,execTime = self._enumerateAlgorithm(algorithm, timeout,verbose, threads)

		if indicator != EnumerationManager.INDICATOR_ALLGOOD:
			if indicator == EnumerationManager.INDICATOR_TIMEOUT:
				self._logger.error('Model Enumeration timed out after: {}s'.format(timeout))
				raise TimeoutException()
			elif indicator == EnumerationManager.INDICATOR_OVERFLOW:
				self._logger.error('Model Enumeration lead to a memory overflow after: {}s'.format(execTime))
				raise OverFlowException()
			else:
				raise Exception("Model enumeartion indicator is: {}".format(indicator))

		return (indicator, list(self._sddManager.getModels()), execTime)

	def _enumerateAlgorithm(self,algorithm, timeout, verbose, threads = 0):
		if self._sddManager == None:
			self._logger.error("Please Initalize the Algorithm first.")
			raise AlgorithmInitException

		self._logger.setVerbose(verbose)

		start_time = time.time()
		indicator = self._executeHavyComputationNEW(self._sddManager, timeout)
		end_time = time.time()

		return indicator, (end_time - start_time)

	def getNodesTraversed(self):
		return self._sddManager.getNodesTraversed()


	'''-----------------------------------------------------------------------------------------'''
	'''----------                      Threading Methods                              ----------'''
	'''-----------------------------------------------------------------------------------------'''


	def _executeHavyComputationNEW(self,sddManager,timeout):
		p = StoppableThreadNEW(sddManager,self._logger)
		p.start()
		sleepTime = 0.001
		increaseTime = True
		start_time = time.time()
		while p.is_alive():
			#print('active count: ',threading.active_count())
			time.sleep(sleepTime)
			tmp_time = time.time()
			# if increaseTime and (tmp_time - start_time) > 10:
			# 	sleepTime = 1
			# 	increaseTime = False
			memoryOverFlow = self.checkMemory()
			timoutExprired = (tmp_time - start_time) > timeout
			if memoryOverFlow or timoutExprired:
				before = threading.active_count()
				self._logger.writeToLog("Setting stop Flag in main Tread: {}".format(threading.current_thread()))
				p.stop()
				self._logger.writeToLog("Joining the thread p now.")
				p.join()
				after = threading.active_count()
				if timoutExprired:
					self._logger.writeToLog("Process {} had to be terminated, since the set timeout of {} hours was reached, threads: before {}, after {}".format(sddManager,timeout/(60 * 60),before,after))
					return EnumerationManager.INDICATOR_TIMEOUT
				elif memoryOverFlow:
					self._logger.writeToLog("Process {}, version had to be terminated, as the Memory Usage overstepped the threshold: {} > {}, threads: before {}, after {}".format(\
						sddManager,psutil.virtual_memory()[2], EnumerationManager.MEMORY_THRESHOLD,before,after))
					return EnumerationManager.INDICATOR_OVERFLOW
				else:
					self._logger.error("Process {}, version had to be terminated as an unkown exeption occured, mem: ({},{}), timout: ({},{})".format(\
						sddManager,psutil.virtual_memory()[2], EnumerationManager.MEMORY_THRESHOLD,(tmp_time - start_time), timeout))
					return EnumerationManager.INDICATOR_UNKNOWN
		p.join()
		return EnumerationManager.INDICATOR_ALLGOOD


	'''-----------------------------------------------------------------------------------------'''
	'''----------                      General Methods                                ----------'''
	'''-----------------------------------------------------------------------------------------'''
	def getOriginalVarCount(self):
		return self._originalVarCount

	def checkMemory(self):
		if psutil.virtual_memory()[2] > EnumerationManager.MEMORY_THRESHOLD:
			self._logger.error('Memory Overflow Execption: {}'.format(psutil.virtual_memory()))
			return True
		else:
			return False

	def freeMemory(self):
		if self._sddManager:
			del self._sddManager
		gc.collect()

class StoppableThreadNEW(threading.Thread):
	"""Thread class with a stop() method. The thread itself has to check
	regularly for the stopped() condition."""
	def __init__(self, algorithm,logger):
		super(StoppableThreadNEW, self).__init__()
		self._stop_event = threading.Event()
		self._algorithm = algorithm
		self._logger = logger

	def stop(self):
		self._stop_event.set()

	def stopped(self):
		return self._stop_event.is_set()

	def run(self):
		#Main exeption handling, as this is the topmost layer of the threading execusion
		try:
			self._algorithm.startAlgorithm(self._stop_event)
		except StopException:
			self._logger.writeToLog("Algorithm is terminated, due to the stop event being set and thread stopped")
			return
		except Exception as a:
			self._logger.error("UNKOWN error caught in running algorithm: {}".format(self._algorithm))
			self._logger.error(traceback.format_exc())
			return





