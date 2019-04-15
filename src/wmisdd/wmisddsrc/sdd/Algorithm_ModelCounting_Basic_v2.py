from .SddBaseManager import SddBaseManager
from .SddStructure import BoolNode, DecNode, LitNode

class Algorithm_ModelCounting_Basic(SddBaseManager):

	# def __init__(self,name,vtreeMan, logger):
	# 	super(Algorithm_ModelCounting_Basic,self).__init__(name,vtreeMan,logger)

	def startAlgorithm(self, stopEvent):
		self._stopEvent = stopEvent

		modelCount = self._getModelCount(self._root)

		vtreeScopeCount = len(self._vtreeMan.getScope(self._vtreeMan.getRoot()))
		if vtreeScopeCount != self._nodes[self._root].scopeCount\
			and not self._isTrue(self._root) and not self._isFalse(self._root):
			missing = vtreeScopeCount - self._nodes[self._root].scopeCount
			modelCount = modelCount * 2**missing
			
		self.setResult(modelCount)
		return

	def _getModelCount(self,nodeId):
		if self._stopEvent.is_set():
			raise Exception("Stop Event set")
			
		node = self._nodes[nodeId]
		if isinstance(node, BoolNode):
			if node.true: 
				return 1
			else: 
				return 0
		elif isinstance(node, LitNode):
			return 1

		modelCount = 0

		for i,(p,s) in enumerate(node.children):

			subScopeCount = self._nodes[s].scopeCount
			primeScopeCount = self._nodes[p].scopeCount
			missing = node.scopeCount - self._nodes[s].scopeCount - self._nodes[p].scopeCount

			tmpCount = self._getModelCount(s) *	 self._getModelCount(p)
			modelCount += tmpCount * 2**(missing)

		return modelCount

