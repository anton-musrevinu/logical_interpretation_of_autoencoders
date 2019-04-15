from .SddBaseManager import SddBaseManager
from .SddStructure import BoolNode, DecNode, LitNode
from BitVector import BitVector
import itertools
from ..elements.myExceptions import StopException


class Algorithm_ModelEnumeration(SddBaseManager):

	def __init__(self,name, logger):
		super(SddBaseManager,self).__init__(name,logger)
		self._processed = 2
		self._stopEvent = None

	'''-----------------------------------------------------------------------------------------'''
	'''----------                 Genreator Specific METHODS                          ----------'''
	'''-----------------------------------------------------------------------------------------'''

	def checkStopEvent(self,info):
		if self._stopEvent.is_set():
			self._logger.error('Raising a timeout or Memory exception from {}.'.format(info))
			raise StopException

	def getModels(self):
		return self.getResult()


	def getNodesTraversed(self):
		return self._processed

	def _getLengthOfGen(self,gen):
		#print('computing the length of this thing')
		saveGen,returnGen = itertools.tee(gen,2)
		return returnGen, sum(1 for x in saveGen)

	def _getGenAsString(self,gen):
		saveGen,returnGen = itertools.tee(gen,2)
		return returnGen, [str(x) for x in saveGen]


	def _productGen(self,input1,input2):
		for i in input1:
			self.checkStopEvent("self._productGen")
			input2,input2copy = itertools.tee(input2,2)
			for j in input2copy:
				yield i | j

	def _productList(self,input1,input2):
		result = []
		for i in input1:
			self.checkStopEvent("self._productList")
			input2,input2copy = itertools.tee(input2,2)
			for j in input2copy:
				result.append(i | j)
		return result

	def _isTrue(self,nodeId):
		return self._nodes[nodeId].isTrue()

	def _isFalse(self,nodeId):
		return self._nodes[nodeId].isFalse()



	def completeModelsGen(self,models,nodeMask,primeId, supId, nodeId = None):
		sMask = self._nodes[supId].scope
		pMask = self._nodes[primeId].scope

		missing = list((set(nodeMask) - set(sMask)) - set(pMask))

		return self._completeModelsGen(models,missing, nodeId)

	def _completeModelsGen(self,models,missing, nodeId = None):
		newModels = []
		table = itertools.product([False, True], repeat=len(missing))

		models, len1 = self._getLengthOfGen(models)
		len2 = 2**len(missing)

		if len2 < len1:
			#print('copying table, {} < {}'.format(len2,len1))
			self._logger.writeToLog('\t\t\t\t-> complete: copying table (T: {},M: {}) for Node: {}'.format(len2,len1, nodeId))
			for model in models:
				self.checkStopEvent("self._completeModelsGen")

				table, tableCopy = itertools.tee(table,2)
				for entry in tableCopy:
					newModel = BitVector(size = self._varFullModelLength)
					for i,truth in enumerate(entry):
						if not missing[i] in self._varMap.keys():
							self._varMap[missing[i]] = self._vtreeMan.getIdOfVariable(missing[i])
						newModel[self._varMap[missing[i]]] = (1 if truth else 0)
					yield model | newModel
		else:
			#print('copying models, {} < {}'.format(len1,len2))
			self._logger.writeToLog('\t\t\t\t-> complete: copying models (T: {},M: {}) for Node: {}'.format(len2,len1, nodeId))
			for entry in table:
				self.checkStopEvent("self._completeModelsGen")

				newModel = BitVector(size = self._varFullModelLength)
				for i,truth in enumerate(entry):
					if not missing[i] in self._varMap.keys():
						self._varMap[missing[i]] = self._vtreeMan.getIdOfVariable(missing[i])
					newModel[self._varMap[missing[i]]] = (1 if truth else 0)
				
				models, modelsCopy = itertools.tee(models,2)
				for model in modelsCopy:
					yield model | newModel

	def completeModelsList(self,models,nodeMask,primeId, supId, nodeId = None):
		sMask = self._nodes[supId].scope
		pMask = self._nodes[primeId].scope

		missing = list((set(nodeMask) - set(sMask)) - set(pMask))

		return self._completeModelsGen(models,missing, nodeId)

	def _completeModelsList(self,models,missing, nodeId = None):
		newModels = []
		table = itertools.product([False, True], repeat=len(missing))

		models, len1 = self._getLengthOfGen(models)
		len2 = 2**len(missing)
		result = []

		if len2 < len1:
			#print('copying table, {} < {}'.format(len2,len1))
			self._logger.writeToLog('\t\t\t\t-> complete: copying table (T: {},M: {}) for Node: {}'.format(len2,len1, nodeId))
			for model in models:
				self.checkStopEvent("self._completeModelsGen")

				table, tableCopy = itertools.tee(table,2)
				for entry in tableCopy:
					newModel = BitVector(size = self._varFullModelLength)
					for i,truth in enumerate(entry):
						if not missing[i] in self._varMap.keys():
							self._varMap[missing[i]] = self._vtreeMan.getIdOfVariable(missing[i])
						newModel[self._varMap[missing[i]]] = (1 if truth else 0)
					result.append(model | newModel)
		else:
			#print('copying models, {} < {}'.format(len1,len2))
			self._logger.writeToLog('\t\t\t\t-> complete: copying models (T: {},M: {}) for Node: {}'.format(len2,len1, nodeId))
			for entry in table:
				self.checkStopEvent("self._completeModelsGen")

				newModel = BitVector(size = self._varFullModelLength)
				for i,truth in enumerate(entry):
					if not missing[i] in self._varMap.keys():
						self._varMap[missing[i]] = self._vtreeMan.getIdOfVariable(missing[i])
					newModel[self._varMap[missing[i]]] = (1 if truth else 0)
				
				models, modelsCopy = itertools.tee(models,2)
				for model in modelsCopy:
					result.append(model | newModel)
		return result

	def writeModelsToFile(self,nodeId,models):
		# if nodeId in self.blockedNodes:
		#     models, leng = self._getLengthOfGen(models)
		#     self.blockedNodes.remove(nodeId)
		#     self._logger.test("WRITE: {}".format(self.blockedNodes))

		models, returnModels = itertools.tee(models,2)
		with open('{}/{}.{}.models'.format(self._pathToModels,self._name,nodeId),'wb') as f:
			for i,model in enumerate(models):
				model.write_to_file(f)
		return returnModels

	def writeModelsToFile_v2(self,nodeId,models):
		# if nodeId in self.blockedNodes:
		#     models, leng = self._getLengthOfGen(models)
		#     self.blockedNodes.remove(nodeId)
		#     self._logger.test("WRITE: {}".format(self.blockedNodes))
		with open('{}/{}.{}.models'.format(self._pathToModels,self._name,nodeId),'wb') as f:
			for model in models:
				model.write_to_file(f)

	def readModelsFromFile(self,nodeId):
		#models = []
		#self.blockedNodes.append(nodeId)
		bvf = BitVector( filename = '{}/{}.{}.models'.format(self._pathToModels,self._name,nodeId) )
		while (bvf.more_to_read):
			yield (bvf.read_bits_from_file(self._varFullModelLength))
		bvf.close_file_object()
		#return models

	def getModelCountFromFile(self,nodeId):
		i = 0
		bvf = BitVector(filename = '{}/{}.{}.models'.format(self._pathToModels,self._name,nodeId))
		while (bvf.more_to_read):
			bvf.read_bits_from_file(self._varFullModelLength)
			i += 1
		#self._logger.writeToLog('ModelCount: i = {}, len(bvf) = {}, len(bvf)/8 = {}'.format(i,len(bvf), len(bvf)/8))
		bvf.close_file_object()
		return i