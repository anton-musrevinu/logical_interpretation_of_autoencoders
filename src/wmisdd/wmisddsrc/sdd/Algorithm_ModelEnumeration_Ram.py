from .Algorithm_ModelEnumeration import Algorithm_ModelEnumeration
from .SddStructure import BoolNode, DecNode, LitNode
from BitVector import BitVector
import itertools

class Algorithm_ModelEnumeration_Ram(Algorithm_ModelEnumeration):

	def __init__(self,name, logger):
		super(Algorithm_ModelEnumeration,self).__init__(name,logger)
		self._processed = 0

	def startAlgorithm(self, stopEvent):
		self._stopEvent = stopEvent

		if self._isFalse(self._root):
			self.setResult([])
			return

		self._processed = 2
		if self._isTrue(self._root):
			models = list([])
		else:
			models = self._getModelsRAM(self._root)

		if self._varFullCount != len(self._varMap):
			mask1 = self._nodes[self._root].scope
			mask2 = self._vtreeMan.getScope(self._vtreeMan.getRoot())
			missing = list(set(mask2) - set(mask1))

			models = self._completeModelsGen(models,missing)


		self.setResult(list(models))
		return

	def _getModelsRAM(self,nodeId):

		self.checkStopEvent("_getModelsRAM, Node: {}".format(nodeId))

		node = self._nodes[nodeId]

		if node.computed:
			modelsSave, modelsReturn = itertools.tee(node.models,2)
			node.models = modelsSave
			return modelsReturn

		elif isinstance(node,LitNode):
			model = BitVector(size = self._varFullModelLength)
			model[node.varId] = (0 if node.negated else 1)

			node.models = list([model])
			node.computed = True
			return node.models

		models = []
		for (p,s) in node.children:
			tmpModels = []
			if self._isFalse(p) or self._isFalse(s):
				continue
			if self._isTrue(p):
				tmpModels = self._getModelsRAM(s)
			elif self._isTrue(s):
				tmpModels = self._getModelsRAM(p)
			else:
				tmpModels = self._productGen(self._getModelsRAM(p),self._getModelsRAM(s))

			#------ complete the computed models
			if node.scopeCount != self._nodes[p].scopeCount + self._nodes[s].scopeCount:
				tmpModels = self.completeModelsGen(tmpModels,node.scope,p, s,nodeId)

			#------ Store the models(to disk or into the node)
			models = itertools.chain(models,tmpModels)
		
		modelsSave, modelsReturn = itertools.tee(models,2)
		node.models = modelsSave
		node.computed = True
		self._logger.writeToLog('\t\t\t [{}%]\tFinished with node: {}, processed: {}/{}'.format((self._processed/self._decNodesNb) * 100,nodeId, self._processed,self._decNodesNb))
		self._processed += 1

		return modelsReturn

	# def getModelCount(self):
	# 	if not self._resultComputed:
	# 		return None
	# 	self._result, modelCount = self._getLengthOfGen(self._result)
	# 	return modelCount

	def getModelCount(self):
		if not self._resultComputed:
			return None
		return len(self._result)

	def getResult(self):
		if not self._resultComputed:
			return None
		else:
			return self._result









