#from z3 import *
import time
import itertools
from .VtreeStructure import Node, Leaf
from .SddStructure import LitNode, BoolNode, DecNode
from BitVector import *
from .VtreeManager import VtreeManager

class SddBaseManager(object):
	"""The Sdd manager, used ot create, read, query and store a sdds structure.

	The Class, gives the methods to hold ONE sdd and the acompaning, vtree
	variables such as depth, and number of nodes.
	It further provides methods such as dfSerach and read to query and build 
	the sdd. 
	This is done, so to save space (keep the actual sdd as small as possible)
	and ease the interaction.
	"""



	def __init__(self,name, logger):
		self._name = name
		self._logger = logger

		self._vtreeMan = None
		self._root = None
		self._decNodesNb = 0
		self._nodes = {}
		self._originalOrder = []
		self._varMap = {}

		self._depth = None

		self._varFullCount = 0
		self._varFullModelLength = 0

		self._resultComputed = False
		self._result = None

		self._processed = 2
		self._stopEvent = None

	'''-----------------------------------------------------------------------------------------'''
	'''----------                   General  SDD Methods                              ----------'''
	'''-----------------------------------------------------------------------------------------'''

	def __str__(self):
		return 'SddManager: {}, root: {}, depth: {}, #nodes: {}, vars: {}, _resultComputed: {}'.format(self._name, self._root, self._depth, len(self._nodes), self._varFullCount, self._resultComputed)

	def initTree(self, vtreeFile, sddFile):
		self._vtreeMan = VtreeManager(self._logger)
		self._vtreeMan.fromFile(vtreeFile)

		self._varFullCount = self._vtreeMan.getVarCount()
		self._varFullModelLength = self._varFullCount + (8 - self._varFullCount % 8)

		self.fromFile(sddFile)
		self._getDepth()

	def _getDepth(self):
		if self._depth == None:
			self._depth = self._nodes[self._root].getDepth(self._nodes)
		return self._depth

	def getVarCount(self):
		return self._varFullCount

	def getNodeCount(self):
		if self._nodes:
			return len(self._nodes)

	def getRoot(self):
		if self._root != None:
			return self._root

	def getSize(self):
		return self._getSize(self._root)

	def _getSize(self,nodeId):
		node = self._nodes[nodeId]
		if isinstance(node, BoolNode):
			return 0
		if isinstance(node, LitNode):
			return 0
		if isinstance(node, DecNode):
			size = 0
			for (p,s) in node.children:
				size += self._getSize(p)
				size += self._getSize(s)
				size += 1
			return size


	'''-----------------------------------------------------------------------------------------'''
	'''----------                  Parser for SDD Methods                      ----------'''
	'''-----------------------------------------------------------------------------------------'''



	def fromFile(self,path):
		"""Reading a file from the SDD package specifide format.

		Reads the vtree from the file given, and stores it as the vtree of 
		this Manager.

		Args:
			path: an absolute path to the .vtree file we want to read
		"""
		#Structure of the file format:
		# c vtree number-of-nodes-in-vtree
		# c L id-of-leaf-vtree-node id-of-variable
		# c I id-of-internal-vtree-node id-of-left-child id-of-right-child

		if path[-4:] != ".sdd":
			self._logger.error("ERROR: Wrong File Format, only .sdd allowed, file given: " + path)
			return


		start_time = time.time()
		file = open(path, "r")

		for i,line in enumerate(file):
			line = line.split(" ")
			id0 = line[0]
			if id0 == "c":
				continue
			elif id0 == "sdd":
				self._nodesNb = int(line[1])
				continue
			#Processing Node
			if not id0 in 'LTFD':
				continue

			id1 = int(line[1]) #id of literal sdd node
			node = None

			if id0 == "L":
				#Syntax:"L id-of-literal-sdd-node id-of-vtree literal"
				id2 = int(line[2]) #id of vtree 
				id3 = line[3] #literal
				var = id3.replace('-',"").replace('\n','')

				if not id2 in self._varMap.keys():
					#My internal id for all variables, build on the idea of models
					varId = int(var) - 1
					self._varMap[id2] = varId
				else:
					varId = self._varMap[id2]

				node = LitNode(id1,id2,varId,id3[0] == '-')

			elif id0 == "T":
				#Syntax:"T id-of-true-sdd-node"
				node = BoolNode(id1,True)
				 #The last element processed has to be the root
			elif id0 == "F":
				#Syntax:"F id-of-false-sdd-node"
				node =  BoolNode(id1,False)
			elif id0 == "D":
				#Syntax:"D id-of-decomposition-sdd-node id-of-vtree 
				#   number-of-elements {id-of-prime id-of-sub}*"
				id2 = int(line[2])
				it = iter([int(x) for x in line[4:]])
				children = []
				try:
					for elem in it:
						sec = next(it)
						primeSdd = self._nodes[elem]
						subSdd = self._nodes[sec]
						if primeSdd == None or subSdd == None:
							self._logger.writeToLog("Child Sdd is none: _nodes[{}]: {}, _nodes[{}]: {}".format(elem,primeSdd,sec,subSdd))
							raise Exception('Could not find child sdd node in _nodes')

						children.append((elem,sec))
			
				except Exception as e:
					self._logger.error("Error caught a layer up: id1: {}, prime: {}".format(id1,elem))
					self._logger.error("Variables: \n\tlen(sdds): {}".format(len(self._nodes)))
					raise e

				node = DecNode(id1,id2,children)
				self._decNodesNb += 1

			if isinstance(node, BoolNode):
				node.setScope({})
			else:
				node.setScope(self._vtreeMan.getScope(id2))
			#self._logger.test('setting scope for ({}) : {}'.format(node.id, node.scope))
			self._nodes[id1] = node
			self._originalOrder.append(id1)
		self._root = self._originalOrder[-1]
		end_time = time.time()

		self._logger.writeToLog("Parsing of the sdd done after {}s, wiht {} Nodes".format(end_time - start_time, self._nodesNb),'debug')

		return 0

	def _isTrue(self,nodeId):
		return self._nodes[nodeId].isTrue()

	def _isFalse(self,nodeId):
		return self._nodes[nodeId].isFalse()


	'''-----------------------------------------------------------------------------------------'''
	'''----------                          ALGORITHM METHODS                          ----------'''
	'''-----------------------------------------------------------------------------------------'''

	def startAlgorithm(self, stopEvent):
		'''Starts the Implemented algorithm, and saves the result somewhere'''
		raise NotImplementedError( "Should have implemented this" )

	def getResult(self):
		if not self._resultComputed:
			return None
		return self._result

	def getModelCount(self):
		return self.getResult()

	def retrieveResult(self):
		if not self._resultComputed:
			self.startAlgorithm()
		return self.getResult()

	def isResultComputed(self):
		return self._resultComputed

	def setResult(self,result):
		if not result == None:
			self._result = result
			self._resultComputed = True





		

