def getModels(self):
	if self._isFalse(self._root):
		return list([])

	if self._isTrue(self._root):
		models = list([])
	else:
		models = self._getModels(self._root)

	if self._varFullCount != len(self._varMap):
		scope1 = self._nodes[self._root].scope
		scope2 = self._vtreeMan.getScope(self._vtreeMan.getRoot())
		missing = list(set(scope1) - set(scope2))
		models = self._completeModels(models,missing)

	return models

def _getModels(self,nodeId):

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
			tmpModels = self._getModels(s)
		elif self._isTrue(s):
			tmpModels = self._getModels(p)
		else:
			tmpModels= self._product(self._getModels(p),\
				self._getModels(s))

		#------ complete the computed models
		if node.scopeCount != self._nodes[p].scopeCount + self._nodes[s].scopeCount:
			tmpModels = self._completeModels(tmpModels,node.scope,p, s)

		models = itertools.chain(models,tmpModels)


	modelsSave, modelsReturn = itertools.tee(node.models,2)
	node.models = modelsSave
	return modelsReturn









