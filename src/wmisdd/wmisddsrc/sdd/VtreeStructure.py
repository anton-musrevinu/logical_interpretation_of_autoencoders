class Vtree(object):
    """The basic vtree parent class, with two children Leaf and Node.

    Each vtree instance can be ether a Leaf or a Node. Both have an id,
    used to reference them.

    Attributes:
        idx: unique index used to identify a spesific Node.
    """
    def __init__(self, idx):
        self.id = idx
        self.scope = {}

    def __str__(self):
        return str(self.id)

class Leaf(Vtree):
    """The Leaf class, a child of Vtree, which hold additionaly a the varId.

    Attributes:
        varId: unique index of the variable used to identify a variable name
    """
    def __init__(self, idx, varId):
        super().__init__(idx)
        self.varId = varId

    def getChildren(self):
        return set()

    # def setScope(self, scope):
    #     self.scope = scope

    # def getNumVars(self):
    #   return 1

    # def findNode(self,idx):
    #   if self.id == idx:
    #       return self

class Node(Vtree):
    """The Node, class, a child of Vtree, which links to its left and right child.

    Attributes:
        left: A Vtree (Node or Leaf), the left child of the node
        right: A Vtree (Node or Leaf), the right child of the node
    """
    def __init__(self, idx, left, right):
        super().__init__(idx)
        self.left = left
        self.right = right

    def getChildren(self):
        return set(self.left,self.right)

    # def setScope(self,nodes):
    #     self.scope = scope
    #     nodes[self.left].setScope(scope[(len(scope)/2)])
