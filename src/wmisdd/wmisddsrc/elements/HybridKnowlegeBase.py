from .Predicate import Predicate
import z3
from BitVector import BitVector

class HybridKnowlegeBase(object):

	def __init__(self, name = None,keep_original_names = False):
		self._name = name
		self._pkbAsString = None
		self._pkbAsZ3 = None
		self._predicates = {}
		self._predicateToIndx = {}
		self._leadVarToPredicateIds = {}
		self._boolVarsToPredicateId = {}
		self._tmpDir = None
		self._nbBoolVars = 0
		self._nbContVars = 0
		self._keep_original_names = keep_original_names

	def is_boolean(self):
		return self._nbContVars == 0 and self._nbBoolVars != 0


	def get_predicate_by_formula(self, formula):
		#ptmp = Predicate(-1, formula)
		return self._predicates[self._predicateToIndx[formula]]

	def contains_formula(self, formula):
		#ptmp = Predicate(-1, formula)
		#print(ptmp.formula, self._predicateToIndx, ptmp.formula in self._predicateToIndx)
		return formula in self._predicateToIndx

	def get_num_of_predicates(self):
		return len(self._predicates)

	def get_continuous_predicate_filter(self):
		realFilter = BitVector(size = self._nbContVars + self._nbBoolVars)
		for idx, p in self._predicates.items():
			realFilter[idx - 1] = p.hasRefinement

		return realFilter

	def store_tmp_dir(self,tmpDir):
		self._tmpDir = tmpDir

	def get_tmp_dir(self):
		return self._tmpDir

	def get_bool_predicate_to_indx(self):
		return self._boolVarsToPredicateId

	def add_predicate(self,originalformula, hasRefinement, boundVariables):
		idx = self.get_num_of_predicates() + 1
		p = Predicate(idx,originalformula, hasRefinement, boundVariables,self._keep_original_names)

		self._predicateToIndx[originalformula] = idx
		self._predicates[idx] = p
		# if p.hasRefinement:
		# 	self._nbContVars += 1
		# 	leadingVar = p.leadingVar
		# 	if leadingVar in self._leadVarToPredicateIds:
		# 		self._leadVarToPredicateIds[leadingVar].append(idx)
		# 	else:
		# 		self._leadVarToPredicateIds[leadingVar] = [idx]
		# else:
		# 	self._nbBoolVars += 1

		return p

	def update_leadVar_dict(self):
		for p in self._predicates.values():
			if p.hasRefinement:
				self._nbContVars += 1
				leadingVar = p.get_leadVar()
				if leadingVar in self._leadVarToPredicateIds:
					self._leadVarToPredicateIds[leadingVar].append(p.id)
				else:
					self._leadVarToPredicateIds[leadingVar] = [p.id]
			else:
				self._nbBoolVars += 1
				self._boolVarsToPredicateId[p.formula] = p.id

	def set_pkb_as_z3(self, pkbAsZ3):
		self._pkbAsZ3 = pkbAsZ3

	def append_pkb_as_z3(self,pkbAsZ3):
		if self._pkbAsZ3 == None:
			self.set_pkb_as_z3(pkbAsZ3)
		else:
			self.set_pkb_as_z3(z3.And(self._pkbAsZ3, pkbAsZ3))

	def append_conditions(self,conditions):
		self.set_pkb_as_z3(z3.simplify(z3.And(self._pkbAsZ3, z3.And(conditions))))

	def get_prop_kb(self):
		return self._pkbAsZ3

	def get_ground_var_refs(self):
		return self._leadVarToPredicateIds

	def get_referenced_predicate_ids(self, leadVar):
		return self._leadVarToPredicateIds[leadVar]

	def get_referenced_predicates(self,leadVar):
		referencedPredicateIds = self.get_referenced_predicate_ids(leadVar)
		refernceddPredicates = map(lambda idx: self.get_predicate(idx), referencedPredicateIds)

		return refernceddPredicates

	def get_predicate(self, key):
		return self._predicates[key]

	def get_predicates_dict(self):
		return self._predicates

	def get_predicates(self):
		return self._predicates.values()


	def __str__(self):
		return_str = 'HKB: '
		if self._pkbAsString != None:
			return_str += self._pkbAsString
		else:
			return_str += str(self.get_prop_kb())
		for k,v in self.get_predicates_dict().items():
			return_str += '\n\t {} -> {}'.format(k,v)
		return return_str








