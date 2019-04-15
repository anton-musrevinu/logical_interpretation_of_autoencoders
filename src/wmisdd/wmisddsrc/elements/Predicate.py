import z3
import math
from .Interval import Interval
from .myExceptions import AbstractionCondition
from .Function import Function
import numpy as np

class Predicate(object):
	"""A frist version of the predicate class"""

	def __init__(self, idx, formula, hasRefinement, boundVariables,keep_original):
		self.id = idx
		self._bound = None

		self.boundVariables = boundVariables
		self.hasRefinement = hasRefinement
		self.formula = formula
		self._newFomula = None
		self._leadVar = None

		if keep_original:
			self._boolref = z3.Bool(str(self.formula))
		else:
			self._boolref = z3.Bool(str(self.id))


	def set_leadVar(self,var):
		self._leadVar = var

	def get_leadVar(self):
		return self._leadVar

	def set_newFormula(self,newFomula):
		self._newFomula = newFomula

	def set_bound(self,bound):
		self._bound = bound

	def get_bound(self):
		return self._bound

	def get_fomula(self):
		return self.formula

	def getBoolRef(self,keep_original = False):
		return self._boolref

	def get_id(self):
		return self.id

	def __str__(self):
		if self._newFomula != None:
			return "({}, {}, {}, {})".format("ID:" + str(self.id),self._newFomula, self.boundVariables, self._bound)
		else:
			return "({}, {}, {}, {})".format("ID:" + str(self.id),self.formula, self.boundVariables, self._bound)

	def getSubVars(self):
		return self.boundVariables

	def get_bound_variables(self):
		return self.boundVariables

	def __eq__(self,other):
		if other == None:
			return False

		return self.id == other.id