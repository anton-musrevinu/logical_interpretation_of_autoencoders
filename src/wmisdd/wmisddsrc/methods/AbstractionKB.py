from ..elements.HybridKnowlegeBase import HybridKnowlegeBase
from ..elements.Predicate import Predicate
from ..elements.Function import WeightFunction
from .AbstractionEssentials import get_bound_vars_in_atom
from ..parsers.ParseInputs import parse_sm1_to_z3
import time

import z3

#-----------------------------------------------   Abstract KB  --------------------------------------------------------

def init_hkb(name, keep_original_names):
	hybridKnowlegeBase = HybridKnowlegeBase(name,keep_original_names)

	return hybridKnowlegeBase

def abstract_hkb(z3Formula, logger, hybridKnowlegeBase = None):

	if hybridKnowlegeBase == None:
		hybridKnowlegeBase = HybridKnowlegeBase()

	propkb = _abstract_hkb_rec(z3Formula, logger, hybridKnowlegeBase)

	hybridKnowlegeBase.set_pkb_as_z3(propkb)
	# logger.writeToLog('HypridKB: {}'.format(hybridKnowlegeBase),'result')
	return hybridKnowlegeBase

def _abstract_hkb_rec(z3Formula, logger, hybridKnowlegeBase):

	op = z3Formula.decl().kind()
	if op in [z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT]:
		return abstraction(hybridKnowlegeBase, z3Formula, logger)
	elif len(z3Formula.children()) > 1:
		decl = z3Formula.decl()
		children = []
		for child in z3Formula.children():
			children.append(_abstract_hkb_rec(child, logger, hybridKnowlegeBase))
		return _return_func(z3Formula, children)
	elif len(z3Formula.children()) == 1:
		abstracted = _abstract_hkb_rec(z3Formula.children()[0], logger, hybridKnowlegeBase)
		return _return_func(z3Formula, [abstracted])
	else:
		return abstraction(hybridKnowlegeBase,z3Formula,logger)

def _return_func(z3Formula, children):
	if z3.is_and(z3Formula):
		return z3.And(*children)
	elif z3.is_or(z3Formula):
		return z3.Or(*children)
	return z3Formula.decl()(*children)

def abstraction(hybridKnowlegeBase,formula,logger):
	# newFormula = z3.simplify(oldformula)
	# if not (z3.is_le(oldformula) or z3.is_lt(oldformula) or z3.is_ge(oldformula) or z3.is_gt(oldformula) or z3.is_const(oldformula)):
	# 	# print('should be * or - or + or /: {}'.format(oldformula))
	# 	return oldformula

	# print('should be le or lt or bool: {}'.format(oldformula))

	# formula = oldformula
	negate = False

	if hybridKnowlegeBase.contains_formula(formula):
		return hybridKnowlegeBase.get_predicate_by_formula(formula).getBoolRef()

	hasRefinement = check_predicate_format(formula)
	boundVariablesInRefinement = get_bound_vars_in_atom(formula, [])
	newPredicate = hybridKnowlegeBase.add_predicate(formula, hasRefinement, boundVariablesInRefinement)
	
	if negate:
		return z3.Not(newPredicate.getBoolRef())
	else:
		return newPredicate.getBoolRef()

def check_predicate_format(formula):
	"""Check if the formula for a given predicate is of the supported format
		Supported formats include: 1. A Bool Ref such as A_x
								   2. An Arithmetic Ref such as a <= x or x <= a where a is number and x is Real/IntRef
								   3. An Arithmetic Ref such as a >= x or x >= a where a is number and x is Real/IntRef
	"""

	if not z3.is_bool(formula):
		raise Exception("Predicate formula is not Bool: {}".format(formula))

	#("Bool formula: {},{},{}, nbGroundAtoms: {}".format(formula, type(formula), formula.children(),nbGroundAtoms))
	if len(formula.children()) == 0:
		#("Boring old Variables such as A_x")
		return False
	if len(formula.children()) >= 2:
		if z3.is_le(formula) or z3.is_ge(formula) or z3.is_lt(formula) or z3.is_gt(formula):
			#("Two or more Children, and <= or => and nbGroundAtoms == 1")
			return True
		else:
			raise Exception("Not supported expression type f: {}, type: {}, nbGroundAtoms: {}".format(\
				formula,type(formula), self.nbGroundAtoms))
	else:
		raise Exception("Predicate formula has an unsupported number of children: {}, type: {}, nbChidren: {}".format(\
			formula,type(formula),formula.children()))
