from .SddBaseManager import SddBaseManager
from .Algorithm_ModelEnumeration import Algorithm_ModelEnumeration
from .SddStructure import BoolNode, DecNode, LitNode
from BitVector import BitVector
import itertools
import threading
from queue import Queue
import traceback
import numpy as np
from shutil import copyfile

class Algorithm_ModelEnumeration_Par(Algorithm_ModelEnumeration):

	VERSION_RAM = "VERSION_RAM"
	VERSION_DISK = "VERSION_DISK"

	def __init__(self,name, logger,threads = 1,pathToModels = None,version = VERSION_RAM):
		super(Algorithm_ModelEnumeration,self).__init__(name,logger)

		if version != self.VERSION_DISK:
			raise Exception('wrong version')

		self._processed = 0
		self._pathToModels = pathToModels		
		if threads < 1:
			raise Exception("Number of desired threads to low: {}".format(threads))
		self._threads = threads
		self._lock = threading.Lock()
		self._queue = None
		self._processedNodes = {}

		self._result = None

	def startAlgorithm(self, stopEvent):

		self._stopEvent = stopEvent

		if self._isFalse(self._root):
			self._logger.writeToLog("\tProblem in UNSAT",'result')
			self.setResult([])
			return


		self._processed = 2
		self._queue = Queue()
		workers = []
		for nodeId in self._nodes.keys():
			self._processedNodes[nodeId] = threading.Event()

		# Create 8 worker threads
		nodesToProcess = 0
		for nodeId in self._originalOrder:
			if not isinstance(self._nodes[nodeId],BoolNode):
				self._queue.put(nodeId)
				nodesToProcess += 1
			else:
				self._processedNodes[nodeId].set()

		tmp = '\t+ starting ME with {} threads to compute {} nodes:'.format(self._threads, nodesToProcess)
		self._logger.writeToLog(tmp,'benchmark')
		if nodesToProcess >= 10:
			self._displayProgressIds = list(map(lambda x: int(x), np.linspace(0,nodesToProcess,10)))
		else:
			self._displayProgressIds = list(range(nodesToProcess))
		# self._logger.writeToLog(str(self._displayProgressIds),'result')
		for x in range(self._threads):

			# tmp += '\n\t\t - starting \'Thread-{}\''.format(x)
			worker = threading.Thread( target=self.runThread, args=('Thread-{}'.format(x),))
			# Setting daemon to True will let the main thread exit even though the workers are blocking
			#worker.daemon = True
			worker.start()
			workers.append(worker)

		# Put the tasks into the queue as a tuple
		# Causes the main thread to wait for the queue to finish processing all the tasks
		if self._stopEvent.is_set():
			self._logger.error('Stop flag is set after queue.join()')
			self.setResult([])
			return
		#print('queue finished, waiting on workers now',workers)
		for w in workers:
			w.join()
		
		models = self.readModelsFromFile(self._root)	
		models, leng = self._getLengthOfGen(models)

		if self._varFullCount != len(self._varMap):
			self._logger.writeToLog('\t\t4.5/5 Finished traversing the tree, completing the models.')
			mask1 = self._vtreeMan.getScope(self._nodes[self._root].vtreeId) #List of Varids
			mask2 = self._vtreeMan.getScope(self._vtreeMan.getRoot())
			missing = list(set(mask2) - set(mask1))
			models = self._completeModelsGen(models,missing)

		# models = list(models)
		# models2 = models[:]
		self.setResult(models)
		return

	def runThread(self,name):
		nodeId = -1
		try:
			while True:
				# Get the work from the queue and expand the tuple
				nodeId = -1
				
				# l = self._lock.acquire()
				if self._queue.empty():
					break
				nodeId = self._queue.get()
				
				self._computeNodeME(nodeId,name, self._processedNodes)
				self._processedNodes[nodeId].set()

				currentProgress = self._queue.qsize() 
				if currentProgress in self._displayProgressIds:
					self._logger.writeToLog('\t\t\t ME  Nodes Progression: {}%'.format(\
						(len(self._displayProgressIds) - self._displayProgressIds.index(currentProgress))\
						/len(self._displayProgressIds) * 100),'benchmark')
				if self._stopEvent.is_set():
					return
		except Exception as e:
			print('ERROR: {},{}'.format(e, traceback.format_exc()))
			self._logger.error("ERROR \"{}\" caught in thread name: {}, {}".format(e, name,threading.currentThread()))
			self._logger.error(traceback.format_exc())
			self._stopEvent.set()
			while True:
				if self._queue.empty():
					break
				t = self._queue.get()
				self._queue.task_done()

			self._logger.error("queue {} is empty now: {}, #threads: {}".format(self._queue ,self._queue.empty(),threading.active_count()))
			return
		return

			#print('====> Setting and notifyieng the lock of {} by thread: {}'.format(nodeId, threading.currentThread()))
			#print(self.queue)

	def _computeNodeME(self,nodeId, threadID, processedNodes):
		#self.findPermutation()
		"""Retruns a list of models that satisry this node.

		Each Model is a dict between varId's and Truth assignments (True, Flase)
		"""
		node = self._nodes[nodeId]

		if isinstance(node,LitNode):
			model = BitVector(size = self._varFullModelLength)
			model[node.varId] = (0 if node.negated else 1)
			# node.models = model
			# return node.models
			self.writeModelsToFile(nodeId,[model])
			return

		if not isinstance(node,DecNode):
			raise Exception("WRONG FORMAT: {}".format(type(node)))

		nodeMask = node.scope
		isfirst = True
		for i,(p,s) in enumerate(node.children):
			if self._isFalse(p) or self._isFalse(s):
				continue
			if self._isTrue(p):
				processedNodes[s].wait()
				tmpModels = s
			elif self._isTrue(s):
				#read prime
				processedNodes[p].wait()
				tmpModels = p
			else:
				processedNodes[p].wait()
				processedNodes[s].wait()
				#read both nodes (ether dec nodes or lit nodes)
				tmpModels = self._productLocal(p,s, str(nodeId) + '_tmp')

			#------ complete the computed models
			# self._logger.writeToLog('updating node {} models with isfirst: {}'.format(nodeId,isfirst),'result')

			if node.scopeCount != self._nodes[p].scopeCount + self._nodes[s].scopeCount:
				self.completeModels(tmpModels,nodeMask,p, s, isfirst,nodeId)
			else:
				src = '{}/{}.{}.models'.format(self._pathToModels,self._name,tmpModels)
				dst = '{}/{}.{}.models'.format(self._pathToModels,self._name,nodeId)

				readas = 'wb' if isfirst else 'ab'
				# self._logger.writeToLog('writing to file: {}, with readas: {}'.format(dst,readas),'result')
				with open(dst,readas) as f:
					bvf1 = BitVector( filename = src )
					while (bvf1.more_to_read):
						model = (bvf1.read_bits_from_file(self._varFullModelLength))
						model.write_to_file(f)
					bvf1.close_file_object()
			isfirst = False

		return


	def _productLocal(self,node1Id, node2Id, tmpNodeId):
		with open('{}/{}.{}.models'.format(self._pathToModels,self._name,tmpNodeId),'wb') as f:
			bvf1 = BitVector( filename = '{}/{}.{}.models'.format(self._pathToModels,self._name,node1Id) )
			while (bvf1.more_to_read):
				bvf2 = BitVector( filename = '{}/{}.{}.models'.format(self._pathToModels,self._name,node2Id) )
				while (bvf2.more_to_read):
					((bvf1.read_bits_from_file(self._varFullModelLength)) \
						| (bvf2.read_bits_from_file(self._varFullModelLength))).write_to_file(f)
				bvf2.close_file_object()
			bvf1.close_file_object()
		return tmpNodeId

	def completeModels(self,modelsID,nodeMask,primeId, supId, isfirst, nodeId):
		sMask = self._nodes[supId].scope
		pMask = self._nodes[primeId].scope

		missing = list((set(nodeMask) - set(sMask)) - set(pMask))

		self._completeModels(modelsID,missing, isfirst, nodeId)

	def _completeModels(self,modelsID,missing, isfirst, nodeId):
		newModels = []
		table = itertools.product([False, True], repeat=len(missing))

		# len1 = self.getModelCountFromFile(modelsID)
		# len2 = 2**len(missing)

		# if len2 < len1:
			#print('copying table, {} < {}'.format(len2,len1))

		readas = 'wb' if isfirst else 'ab'

		# self._logger.writeToLog('writing to file: {}, with readas: {}'.format(dst,readas),'result')
		with open('{}/{}.{}.models'.format(self._pathToModels,self._name,nodeId),readas) as f:
			bvf1 = BitVector( filename = '{}/{}.{}.models'.format(self._pathToModels,self._name,modelsID) )
			while (bvf1.more_to_read):
				self.checkStopEvent("self._completeModels")
				model = (bvf1.read_bits_from_file(self._varFullModelLength))

				table, tableCopy = itertools.tee(table,2)
				for entry in tableCopy:
					newModel = BitVector(size = self._varFullModelLength)
					for i,truth in enumerate(entry):
						if not missing[i] in self._varMap.keys():
							self._varMap[missing[i]] = self._vtreeMan.getIdOfVariable(missing[i])
						newModel[self._varMap[missing[i]]] = (1 if truth else 0)
					(model | newModel).write_to_file(f)
			bvf1.close_file_object()
		# else:
		# 	#print('copying models, {} < {}'.format(len1,len2))
		# 	self._logger.writeToLog('\t\t\t\t-> complete: copying models (T: {},M: {}) for Node: {}'.format(len2,len1, nodeId))
		# 	for entry in table:
		# 		self.checkStopEvent("self._completeModelsGen")

		# 		newModel = BitVector(size = self._varFullModelLength)
		# 		for i,truth in enumerate(entry):
		# 			if not missing[i] in self._varMap.keys():
		# 				self._varMap[missing[i]] = self._vtreeMan.getIdOfVariable(missing[i])
		# 			newModel[self._varMap[missing[i]]] = (1 if truth else 0)
				
		# 		models, modelsCopy = itertools.tee(models,2)
		# 		for model in modelsCopy:
		# 			yield model | newModel

	def getModelCount(self):
		if not self._resultComputed:
			return None

		if self._version == Algorithm_ModelEnumeration_Par.VERSION_DISK:
			modelCount = self.getModelCountFromFile(self._root)
		elif self._version == Algorithm_ModelEnumeration_Par.VERSION_RAM:
			self._result, modelCount = self._getLengthOfGen(self._result)

		return modelCount

	def setResult(self,result):

		modelCount = self.writeModelsToFile(self._root,result)

		self._resultComputed = True
	
	def getResult(self):
		if not self._resultComputed:
			return None

		if self._result != None:
			models = self._result
		else:
			models = self.readModelsFromFile(self._root)

		return models









