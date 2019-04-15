from ..elements.Predicate import Predicate
from ..elements.Function import WeightFunction
from .AbstractionEssentials import mapFunctionSymbol

import z3

#-----------------------------------------------   Abstract KB  --------------------------------------------------------

def _abstact_wf_rec(z3Formula, logger, weightFunction):
	op = z3Formula.decl().kind()
	if len(z3Formula.children()) == 0 and (z3.is_real(z3Formula) or z3.is_int(z3Formula) or z3.is_bool(z3Formula)) \
			and not (z3.is_int_value(z3Formula) or z3.is_rational_value(z3Formula)):
		if not weightFunction.contains_variable(z3Formula):
			weightFunction.add_variable(z3Formula)
		return z3Formula
	else:
		children = []
		for child in z3Formula.children():
			children.append(_abstact_wf_rec(child, logger, weightFunction))
		return _return_func(z3Formula, children)

def _return_func(z3Formula, children):
	if z3.is_and(z3Formula):
		return z3.And(*children)
	elif z3.is_or(z3Formula):
		return z3.Or(*children)
	return z3Formula.decl()(*children)

def abstract_single_wf(wfAsZ3, logger):

	weightFunction = WeightFunction()

	wfAsZ3_abs = _abstact_wf_rec(wfAsZ3, logger, weightFunction)

	logger.writeToLog('finished parsing wf: {}'.format(wfAsZ3_abs),'info')
	weightFunction.set_function_as_z3(wfAsZ3_abs)
	
	return weightFunction


# def abstract_single_wf(wString, logger):

# 	weightFunction = WeightFunction()

# 	stack = list()
# 	functionSymbols = '&|*+-<=>=~^ite'
# 	stringVariables = []
# 	stringVariablesBool = []
# 	string = ""
# 	try:
# 		string = wString.replace('\n','').split('(')
# 		idx = 0
# 		for pos,elem in enumerate(string):
# 			logger.writeToLog('{} - reading :{}:'.format(pos,elem))
# 			if elem == '':
# 				continue

# 			elemL = elem.split()
# 			# ALL lines should start with '('
# 			formula = None
# 			if elemL[0] in functionSymbols:
# 				formula = elemL[0]
# 			elif elemL[0] == 'var':
# 				varName = elemL[2].replace(')','')
# 				if elemL[1] == 'bool':
# 					formula = z3.Bool(varName)
# 				elif elemL[1] == 'real':
# 					formula = z3.Real(varName)
# 				elif elemL[1] == 'int':
# 					formula = z3.Int(varName)
# 				else:
# 					raise Exception('Unknown Variable format: {}'.format(elemL))

# 				if not weightFunction.contains_variable(formula):
# 					weightFunction.add_variable(formula)	

# 			elif elemL[0] == 'const':
# 				const = elemL[2].replace(')','')
# 				if elemL[1] == 'real':
# 					formula = z3.RealVal(const)
# 				elif elemL[1] == 'int':
# 					formula = z3.IntVal(const)
# 				else:
# 					raise Exception('Unknown Constant format: {}'.format(elemL))
# 			else:
# 				logger.error("Unknown sequence : {}\n\t Occured parsing formula: {}".format(elemL[0],string))
# 				raise Exception

# 			stack.append(formula)

# 			closedBrackets = elem.count(')') - 1
# 			logger.writeToLog("{} - new element in stack: {}\t,cB {}".format(pos ,stack[-1],closedBrackets))
			
# 			if closedBrackets < 1:
# 				continue

# 			while closedBrackets > 0:
# 				logger.writeToLog('{} - stack: {},{}'.format(pos,stack, closedBrackets))
# 				tmpPredi = []
# 				pred = None
# 				while True:
# 					pred = stack.pop()
# 					if isinstance(pred,float) or isinstance(pred,int) or z3.is_int(pred) or z3.is_real(pred) or z3.is_bool(pred):
# 						tmpPredi.append(pred)
# 					else:
# 						break

# 				if len(tmpPredi) == 1:
# 					tmpPredi = tmpPredi[0]
# 					stack.append(mapFunctionSymbol(str(pred))(tmpPredi))
# 				else:
# 					tmpPredi = tmpPredi[::-1]
# 					stack.append(mapFunctionSymbol(str(pred))(tmpPredi))

# 				logger.writeToLog('{} - {} is applied to {}'.format(pos, pred,tmpPredi))


# 				logger.writeToLog("{} - new element in stack: {}\t,cB {}".format(pos ,stack[-1],closedBrackets))
# 				closedBrackets -= 1

# 			logger.writeToLog('{} - finished :{}:'.format(pos,stack))
# 	except Exception as e:
# 		logger.writeToLog("Some Error : {}\n\t Occured parsing formula: {}".format(e,wString))
# 		raise Exception

# 	if len(stack) != 1:
# 		raise Exception("Parsing Error, stack != 1, stack: {}".format(stack))

# 	# functionBodyAsString = stack[0]
# 	# weightFunction.set_function_body_as_string(functionBodyAsString)
# 	logger.writeToLog('finished parsing wf: {}'.format(stack[0]),'info')
# 	weightFunction.set_function_as_z3(stack[0])
	
# 	return weightFunction




# def _createFunctionForSymbol(funcSym,variables):
# 	arithmeticFucntions = "*+&|-<=>="
# 	singletonFunctions = '~'
# 	doubleFunctions = '^'
# 	ifelseStatement = 'ite'
# 	if funcSym in arithmeticFucntions:
# 		term = '( ' + variables[0] + ' '
# 		for var in variables[1::]:
# 			term += _mapFunctionSymbol(funcSym) + ' ' + var + ' '
# 		term += ")"
# 	elif funcSym in singletonFunctions and len(variables) == 1:
# 		term = '{}({})'.format(_mapFunctionSymbol(funcSym),variables[0])
# 	elif funcSym in doubleFunctions and len(variables) == 2:
# 		term = '({}{}{})'.format(variables[0],_mapFunctionSymbol(funcSym),variables[1])
# 	elif funcSym in ifelseStatement and len(variables) == 3:
# 		term = '({} if {} else {})'.format(variables[1], variables[0],variables[2])
# 	else:
# 		raise Exception("WRONG SYMBOL FOUND : {}, len: {}, type: {}".format(funcSym,len(variables), type(variables)))
# 	return term

# def _mapFunctionSymbol(symbol):
# 	if symbol == '&':
# 		return 'and'
# 	elif symbol == '|':
# 		return 'or'
# 	elif symbol == '~':
# 		return 'not'
# 	elif symbol == '^':
# 		return '**'
# 	elif symbol in '- * + < <= > >=':
# 		return symbol
# 	else:
# 		raise Exception("WRONG SYMBOL FOUND : {}".format(symbol))