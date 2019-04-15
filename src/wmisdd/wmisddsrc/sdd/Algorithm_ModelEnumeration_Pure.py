from .Algorithm_ModelEnumeration import Algorithm_ModelEnumeration
from .SddStructure import BoolNode, DecNode, LitNode
from BitVector import BitVector
import itertools

class Algorithm_ModelEnumeration_Pure(Algorithm_ModelEnumeration):

	def __init__(self,name, logger):
		super(Algorithm_ModelEnumeration,self).__init__(name,logger)

	def startAlgorithm(self, stopEvent):
		self._stopEvent = stopEvent

		if self._isFalse(self._root):
			self.setResult([])
			return

		self._processed = 2
		if self._isTrue(self._root):
			models = list([])
		else:
			models = self._getModelsPure(self._root)

		if self._varFullCount != len(self._varMap):
			mask1 = self._nodes[self._root].scope
			mask2 = self._vtreeMan.getScope(self._vtreeMan.getRoot())
			missing = list(set(mask2) - set(mask1))
			models = self._completeModelsGen(models,missing)

		self.setResult(list(models))
		return

	def _getModelsPure(self,nodeId):

		self.checkStopEvent("_getModelsPure, Node: {}, processed: {}".format(nodeId, self._processed))

		node = self._nodes[nodeId]

		if isinstance(node,LitNode):
			model = BitVector(size = self._varFullModelLength)
			model[node.varId] = (0 if node.negated else 1)
			nodeModels = list([model])
			return nodeModels

		nodeModels = []
		for i,(p,s) in enumerate(node.children):
			tmpModels = []
			if self._isFalse(p) or self._isFalse(s):
				continue
			if self._isTrue(p):
				#read sup
				tmpModels = self._getModelsPure(s)
			elif self._isTrue(s):
				#read prime
				tmpModels = self._getModelsPure(p)
			else:
				#read both nodes (ether dec nodes or lit nodes)
				tmpModels= self._productGen(self._getModelsPure(p),\
					self._getModelsPure(s))

			#------ complete the computed models
			if node.scopeCount != self._nodes[p].scopeCount + self._nodes[s].scopeCount:
				tmpModels = self.completeModelsGen(tmpModels,node.scope,p, s)

			#------ Store the models(to disk or into the node)
			nodeModels = itertools.chain(nodeModels,tmpModels)
			tmpModels = []

		self._logger.writeToLog('\t\t\t [{}%]\tFinished with node: {}, processed: {}/{}'.format((self._processed/self._decNodesNb) * 100,nodeId, self._processed,self._decNodesNb))
		self._processed += 1
		return list(nodeModels)

	def getModelCount(self):
		if not self._resultComputed:
			return None
		return len(self._result)









