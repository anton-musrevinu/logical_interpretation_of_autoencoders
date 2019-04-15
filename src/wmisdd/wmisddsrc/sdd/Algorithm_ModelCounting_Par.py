from .SddBaseManager import SddBaseManager
from .SddStructure import BoolNode, DecNode, LitNode
import threading
from queue import Queue

class Algorithm_ModelCounting_Par(SddBaseManager):

	def __init__(self,name, logger, threads = 1):
		super(Algorithm_ModelCounting_Par,self).__init__(name,logger)
		if threads < 1:
			raise Exception("Number of desired threads to low: {}".format(threads))
		self._threads = threads

	def startAlgorithm(self, stopEvent):
#def computeModelCountRec(self,stoppedFunc = None,threads = None):

		if self._isFalse(self._root):
			self._result = 0
			self._nodes[self._root].modelCount = 0
			return

		self._logger.resetProgress()
		self._modelCounts = {}
		queue = Queue()
		processed = {}
		for nodeId in self._nodes.keys():
			processed[nodeId] = threading.Event()
		# Create 8 worker threads
		for x in range(self._threads):
			worker = McWorker(queue,processed,self._nodes, self._modelCounts,stopEvent,self._logger)
			# Setting daemon to True will let the main thread exit even though the workers are blocking
			worker.daemon = True
			worker.start()
		# Put the tasks into the queue as a tuple
		for nodeId in self._originalOrder:
			#self._logger.test('Queueing {}'.format(nodeId))
			queue.put(nodeId)
		# Causes the main thread to wait for the queue to finish processing all the tasks
		queue.join()

		if not self._root in self._modelCounts:
			return

		modelCount = self._modelCounts[self._root]

		if len(self._vtreeMan.getScope(self._vtreeMan.getRoot())) != self._nodes[self._root].scopeCount:
			mask1 = self._nodes[self._root].scope #List of Varids
			mask2 = self._vtreeMan.getScope(self._vtreeMan.getRoot())
			missing = len(list(set(mask2) - set(mask1)))
			modelCount = modelCount * 2**missing

		self.setResult(modelCount)

	def _getModelCount(self,nodeId):
		node = self._nodes[nodeId]
		if isinstance(node, BoolNode):
			if node.true:
				return 1
			else:
				return 0
		elif isinstance(node, LitNode):
			return 1
		elif isinstance(node, DecNode) and node.models:
			return node.modelCount

		node.modelCount = 0
		nodeMaskCount = node.scopeCount
		for i,(p,s) in enumerate(node.children):
			sM = self._getModelCount(s)
			if sM == 0:
				continue
			pM = self._getModelCount(p)
			if pM == 0:
				continue

			tmpModelCount = pM * sM

			sMaskCount = self._nodes[s].scopeCount#self._vtreeMan.getScope(self._nodes[s].vtreeId)
			pMaskCount = self._nodes[p].scopeCount#self._vtreeMan.getScope(self._nodes[p].vtreeId)
			if nodeMaskCount != sMaskCount + pMaskCount:
				missing = nodeMaskCount - sMaskCount - pMaskCount
				tmpModelCount = tmpModelCount * 2**missing

			node.modelCount += tmpModelCount

		return node.modelCount

class McWorker(threading.Thread):
	def __init__(self, queue,processed,nodes,modelCounts,stopEvent,logger):
		threading.Thread.__init__(self)
		self.queue = queue
		self.processed = processed
		self._nodes = nodes
		self.modelCounts = modelCounts
		self._stopEvent = stopEvent
		self._logger = logger

	def run(self):
		try:
			while True:
				# Get the work from the queue and expand the tuple
				nodeId = self.queue.get()
				#print('----> Starting on node:\t {} \tby thread: {}'.format(nodeId, threading.currentThread()))
				modelCounts = self._computeNodeMC(nodeId)
				self.modelCounts[nodeId] = modelCounts
				self.processed[nodeId].set()
				self.queue.task_done()
		except Exception as e:
			self._logger.error("ERROR \"{}\" caught in thread {}".format(e, threading.currentThread()))
			self._logger.error(traceback.format_exc())
			self._stopEvent.set()
			return

	def _computeNodeMC(self,nodeId):
		if self._stopEvent.is_set():
			raise Exception("Stop Event Set")
		node = self._nodes[nodeId]

		if isinstance(node, BoolNode):
			return (1 if node.true else 0)
		elif isinstance(node,LitNode):
			return 1

		tmp_node_mc = 0
		nodeMaskCount = node.scopeCount
		for i, (p,s) in enumerate(node.children):
			self.processed[s].wait()
			if self.modelCounts[s] == 0: continue
			self.processed[p].wait()
			if self.modelCounts[p] == 0: continue

			tmp_mc = self.modelCounts[s] * self.modelCounts[p]

			sMaskCount = self._nodes[s].scopeCount
			pMaskCount = self._nodes[p].scopeCount

			missing = nodeMaskCount - sMaskCount - pMaskCount
			tmp_mc = tmp_mc * 2**missing

			tmp_node_mc += tmp_mc

		return tmp_node_mc