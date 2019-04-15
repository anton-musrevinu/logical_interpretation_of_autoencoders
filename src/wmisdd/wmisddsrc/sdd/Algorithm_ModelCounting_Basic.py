from .SddBaseManager import SddBaseManager
from .SddStructure import BoolNode, DecNode, LitNode

class Algorithm_ModelCounting_Basic(SddBaseManager):

	def __init__(self,name, logger):
		super(Algorithm_ModelCounting_Basic,self).__init__(name,logger)
		self._processed = 0

	def startAlgorithm(self, stopEvent):
#def computeModelCountRec(self,stoppedFunc = None,threads = None):
		self._stopEvent = stopEvent

		if self._isFalse(self._root):
			self.setResult(0)
			return

		modelCount = self._getModelCount(self._root)

		if len(self._vtreeMan.getScope(self._vtreeMan.getRoot())) != self._varFullCount:
			raise Exception('VARIABLE NUMBER IS NOT MATCHING')

		if len(self._vtreeMan.getScope(self._vtreeMan.getRoot())) != self._nodes[self._root].scopeCount:
			mask1 = self._nodes[self._root].scope #List of Varids
			mask2 = self._vtreeMan.getScope(self._vtreeMan.getRoot())
			missing = len(list(set(mask2) - set(mask1)))
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
		elif isinstance(node, DecNode) and node.modelCount != -1:
			return node.modelCount

		node.modelCount = 0
		for (p,s) in node.children:

			subScopeCount = self._nodes[s].scopeCount
			primeScopeCount = self._nodes[p].scopeCount
			missing = node.scopeCount - self._nodes[s].scopeCount - self._nodes[p].scopeCount

			tmpCount = self._getModelCount(s) *	 self._getModelCount(p)
			node.modelCount += tmpCount * 2**(missing)

		self._processed += 1
		#self._logger.writeToLog("Finished processing Node: {}, {}/{}".format(nodeId, self._processed, len(self._originalOrder)),'info')
		return node.modelCount





