class Sdd(object):
    """The basic vtree parent class, with two children Leaf and Node.

    Each vtree instance can be ether a Leaf or a Node. Both have an id,
    used to reference them.

    Attributes:
        idx: unique index used to identify a spesific Node.
    """
    def __init__(self, idx):
        self.id = idx
        self.scope = {}
        self.scopeCount = 0
        self.computed = False
        self.models = None

    def __str__(self):
        return str(self.id)

    def cleanNode(self):
        self.computed = False
        self.models = False

    def setScope(self,scope):
        self.scope = scope
        self.scopeCount = len(scope)

    def isTrue(self):
        return False

    def isFalse(self):
        return False

class LitNode(Sdd):
    """The LitNode class, a child of Sdd represents a literal sdd node.

    Attributes:
        vtreeId: unique index to reference the Vtree node, this node 
            corresponds to
        varId: unique index of the variable used to identify a variable name
    """
    def __init__(self, idx, vtreeId, varId, negated):
        super().__init__(idx)
        self.vtreeId = vtreeId
        self.varId = varId
        self.negated = negated

    def getFunction(self):
        if self.negated:
            return Not(Bool(str(self.varId)))
        else:
            return Bool(str(self.varId))

    def getModels(self,nodes,processed,totalNb,verbose = None):
        model = BitVector()
        if self.negated:
            return list([BoolNode()]), processed
        else:
            return list([{self.varId : True}]), processed

    # def getModelCount(self,vtree, verbose = None):
    #   return 1

    def getChildren(self):
        return []

    def getDepth(self,nodes):
        return 0

    def __str__(self):
        if self.models != None:
            return 'LitNode: id: {}, vtreeId: {}, varId: {}, negated: {}, scope: {}, model: {}'.format(\
                self.id,self.vtreeId, self.varId,self.negated,self.scope, self.models[0])
        else:
            return 'LitNode: id: {}, vtreeId: {}, varId: {}, negated: {}, scope: {}, model: {}'.format(\
                self.id,self.vtreeId, self.varId,self.negated,self.scope, self.models)

class BoolNode(Sdd):
    """The BoolNode class, a child of Sdd represents a boolean sdd node.

    Attributes:
        true: A boolean variable inidcating wheter this a T of F node
    """
    def __init__(self, idx, true):
        super().__init__(idx)
        self.true = true

    def getFunction(self):
        return self.true

    def isTrue(self):
        return self.true

    def isFalse(self):
        return not self.true

    # def getModelCount(self,vtree, verbose = None):
    #   if self.true:
    #       return -1
    #   else:
    #       return 0

    def getModels(self,nodes,processed,totalNb,verbose = None):
        if self.true:
            return [], processed
        else:
            return None, processed

    def getChildren(self):
        return []

    def getDepth(self,nodes):
        return 0

class DecNode(Sdd):
    """The DecNode class, a child of Sdd represents a decomposition Sdd node.

    Attributes:
        vtreeId: unique idex to reference the vtree node, tis node
            corresponds to
        children: a list of tuples of prime, sub pairs, (non-empty)
    """

    def __init__(self, idx, vtreeId, children):
        super().__init__(idx)
        self.vtreeId = vtreeId
        if(not children):
            raise Exception("ERROR: Children can't be emtpy")
        self.children = children
        self.numChild = len(children)
        self.depth = None
        self.modelCount = -1

    def cleanNode(self):
        self.computed = False
        self.depth = None
        self.models = None
        self.modelCount = -1

    def getFunction(self):
        f = []
        for ps in self.children:
            f.append(And(ps[0].getFunction(),ps[1].getFunction()))
        return Or(f)

    def getChildren(self):
        children = []
        for child in self.children:
            children.append(str(child[0]))
            children.append(str(child[1]))
        return children

    def getDepth(self,nodes):
        if self.depth:
            return self.depth
        else:
            tmp = [max(nodes[child[0]].getDepth(nodes), nodes[child[1]].getDepth(nodes)) for child in self.children]    
            nodes[self.id].depth = max(tmp) + 1
            return self.depth
        