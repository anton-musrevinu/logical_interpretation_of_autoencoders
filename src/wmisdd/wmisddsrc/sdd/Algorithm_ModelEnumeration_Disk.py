from .Algorithm_ModelEnumeration import Algorithm_ModelEnumeration
from .SddStructure import BoolNode, DecNode, LitNode
from BitVector import BitVector
import itertools

class Algorithm_ModelEnumeration_Disk(Algorithm_ModelEnumeration):

	def __init__(self,name, logger,pathToModels):
		super(Algorithm_ModelEnumeration,self).__init__(name,logger)
		self._pathToModels = pathToModels
		self._unsat = False

	def startAlgorithm(self, stopEvent):
		self._stopEvent = stopEvent

		if self._isFalse(self._root):
			self._unsat = True
			self._resultComputed = True
			return

		if self._isTrue(self._root):
			models = list([])
		else:		
			models = self._getModelsDisk(self._root)

		if self._varFullCount != len(self._varMap):
			mask1 = self._nodes[self._root].scope
			mask2 = self._vtreeMan.getScope(self._vtreeMan.getRoot())
			missing = list(set(mask2) - set(mask1))

			models = self._completeModelsGen(models,missing)
			models, leng = self._getLengthOfGen(models)
			self.writeModelsToFile(self._root,models)
			
		self._resultComputed = True

	def _getModelsDisk(self,nodeId):
		#self.findPermutation()
		"""Retruns a list of models that satisry this node.

		Each Model is a dict between varId's and Truth assignments (True, Flase)
		"""
		self.checkStopEvent("_getModelsDisk, Node: {}".format(nodeId))

		node = self._nodes[nodeId]

		if isinstance(node,LitNode):
			if node.computed:
				return node.models
			model = BitVector(size = self._varFullModelLength)
			model[node.varId] = (0 if node.negated else 1)
			node.models = list([model])
			node.computed = True

			return node.models
		elif node.computed:
			return self.readModelsFromFile(nodeId)

		nodeModels = []
		for (p,s) in node.children:
			tmpModels = []
			if self._isFalse(p) or self._isFalse(s):
				continue
			if self._isTrue(p):
				tmpModels = self._getModelsDisk(s)
			elif self._isTrue(s):
				#read prime
				tmpModels = self._getModelsDisk(p)
			else:
				#read both nodes (ether dec nodes or lit nodes)
				tmpModels= self._productGen(self._getModelsDisk(p),\
					self._getModelsDisk(s))

			#------ complete the computed models
			if node.scopeCount != self._nodes[p].scopeCount + self._nodes[s].scopeCount:
				tmpModels = self.completeModelsGen(tmpModels,node.scope,p, s)

			#------ Store the models(to disk or into the node)
			nodeModels = itertools.chain(nodeModels,tmpModels)
			tmpModels = []

		#------ return the models
		returnModels = self.writeModelsToFile(nodeId,nodeModels)
		node.computed = True

		self._logger.writeToLog('\t\t\t [{}%]\tFinished with node: {}, processed: {}/{}'.format((self._processed/self._decNodesNb) * 100,nodeId, self._processed,self._decNodesNb))

		self._processed += 1
		return returnModels


	def getModelCount(self):
		if not self._resultComputed:
			return None
		if self._unsat:
			return 0
		else:
			return self.getModelCountFromFile(self._root)
	
	def getResult(self):
		if not self.isResultComputed():
			return None
		if self._unsat:
			return []
		else:
			return self.readModelsFromFile(self._root)









