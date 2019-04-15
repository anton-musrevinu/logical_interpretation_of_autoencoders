from .VtreeStructure import Vtree, Leaf, Node

class VtreeManager:
	"""The Vtree manager, used ot create, read and store a vtree structure.

	The Class, gives the methods to hold the ONE vtree and the acompaning
	variables such as depth, and number of nodes.
	It further provides methods such as search and read to query and build 
	the vtree. 
	This is done, so to save space (keep the actual vtree as small as possible)
	and ease the interaction.
	"""
	def __init__(self,logger):
		self._vars = []
		self._depth = None
		self._nodesNb = 0
		self._nodes = {}
		self._root = 0
		self._varsOrdered = None
		self._logger = logger


	def fromFile(self,path,verbose = False):
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
		
		if path[-6:] != ".vtree":
			self._logger.debug("ERROR: Wrong File Format, only .vtree allowed, file given: " + path)
			return

		file = open(path, "r") 
		vStack = {}

		for line in file:
			line = line.split(" ")
			id0 = line[0]
			if id0 == "c":
				continue
			if id0 == "vtree":
				self._nodesNb = int(line[1])
				continue
			elif id0 == "L":
				id1 = int(line[1])
				id2 = int(line[2])
				#vStack[id1] = Leaf(id1,id2)
				self._nodes[id1]  = Leaf(id1,id2)
				self._vars.append(id2)
				self._root = id1
			elif id0 == "I":
				id1 = int(line[1])
				id2 = int(line[2])
				id3 = int(line[3]) 

				#tmp = Node(id1, vStack[id2],vStack[id3])
				self._nodes[id1] = Node(id1, id2,id3)
				self._root = id1
				#del vStack[id2]
				#del vStack[id3]
		if verbose:
			self._logger.debug('DEBUG: Reading with Root: ({},{})'.format(self._root, type(self._root)))
		#self._logger.debug("Parsing of the vtree done, wiht depth = {} and {} Nodes".format(self._depth, self._nodes))
		self._computeStuff()

	def getDepth(self):
		if not self._depth:
			self._depth = self._depthrec(self._root)
		return self._depth

	def getVarCount(self):
		return len(self._vars)
		
	def _depthrec(self,node):
		vtree = self._nodes[node]
		if isinstance(vtree,Leaf):
			return 0
		elif isinstance(vtree,Node):
			return max(self._depthrec(vtree.left),self._depthrec(vtree.right)) + 1

	def getIdOfVariable(self,vtreeId):
		return self._nodes[vtreeId].varId - 1



	# def printvTree(self):
	# 	depth = self._depth
	# 	self._logger.debug("The final thing:")
	# 	self._logger.debug(self._self._logger.debug(depth,list([self._vtree])))

	# def _self._logger.debug(self,depth, vtrees):
	# 	#self._logger.debug("Next on" + str(depth))
	# 	out = depth * "\t"
	# 	vtreesNew = []
	# 	for vtree in vtrees:
	# 		if isinstance(vtree,Leaf):
	# 			if not vtree.varId in self._varMap:
	# 				self._logger.debug("VARIABLE MAPPING COULD NOT BE FOUND")
	# 			out = out + "  " + self._varMap[vtree.varId] + "[" + str(vtree) + "]\t"
	# 		else:
	# 			out = out + str(vtree) + "\t" * (depth + 1)
	# 			#self._logger.debug("2" + out)
	# 			vtreesNew.append(vtree.left)
	# 			vtreesNew.append(vtree.right)
	# 	out = out + "\n"
	# 	#self._logger.debug("3" + out)
	# 	if not vtrees:
	# 		return out
	# 	else:
	# 		#self._logger.debug("4" + str(vtreesNew))
	# 		return out + self._self._logger.debug(depth -1, vtreesNew)

	def isBalanced(self):
		root = self._nodes[self._root]
		if isinstance(root,Leaf):
			return False

		if(abs(self._depthrec(root.left) - self._depthrec(root.right)) <= 1):
			return True
		else:
			return False

	def getVtree(self):
		return self._root, self._nodes

	def getVarsOrdered(self):
		if self._varsOrdered:
			return self._varsOrdered
		else:
			self._computeVarOrder(self._root, True)
			return self._varsOrdered

	def _computeVarOrder(self,node,storeScope = None):
		if not self._varsOrdered:
			self._varsOrdered = set()
		node = self._nodes[node]
		if isinstance(node, Leaf):
			self._varsOrdered.add(node.id)
			if storeScope:
				node.scope = [node.id]
			return [node.id]
		elif isinstance(node,Node):
			left = self._computeVarOrder(node.left,storeScope)
			right = self._computeVarOrder(node.right,storeScope)
			if storeScope:
				node.scope = self._nodes[node.left].scope + self._nodes[node.right].scope
			return left + right

	def doCheckScope(self,verbose = False):
		self._computeVarOrder(self._root, storeScope = True)
		self._checkScopes(self._root,verbose)

	def _checkScopes(self,nodeId,verbose = False):
		node = self._nodes[nodeId]
		if node.scope == self._computeVarOrder(nodeId, storeScope = False):
			if verbose:
				self._logger.debug('NodeId: {} -> scope: {}'.format(nodeId,node.scope))
			if isinstance(node,Node):
				self._checkScopes(node.left,verbose)
				self._checkScopes(node.right,verbose)
		else:
			self._logger.debug("not equal pair found: \nnode: {} with \n{} and {}".format(nodeId,node._scope, self._computeScope(nodeId)))

	def getRoot(self):
		return self._root

	def getScope(self,nodeId):
		return set(self._nodes[nodeId].scope)

	def _computeStuff(self):
		self.getDepth()
		self._computeVarOrder(self._root, storeScope = True)


	# def _computeScope(self):





	# def getNumVars(self):
	# 	return self.left.getNumVars() + self.right.getNumVars()


	# def findNode(self, idx):
	# 	#self._logger.debug('Searching node {} for {}, {}'.format(self,idx,self.id == idx))
	# 	if self.id == idx:
	# 		return self
	# 	else:
	# 		tmp = self.left.findNode(idx)
	# 		if tmp != None:
	# 			return tmp
	# 		else:
	# 			return self.right.findNode(idx)


