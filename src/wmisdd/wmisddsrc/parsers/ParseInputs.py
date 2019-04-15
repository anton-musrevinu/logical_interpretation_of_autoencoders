import z3
import functools


def parse_sm1_to_z3(wString,logger):
	stack = list()
	functionSymbols = '&|*+-<=>=~^ite'
	stringVariables = []
	stringVariablesBool = []
	string = ""
	try:
		string = wString.replace('\n','').split('(')
		idx = 0
		for pos,elem in enumerate(string):
			logger.writeToLog('{} - reading :{}:'.format(pos,elem))
			if elem == '':
				continue

			elemL = elem.split()
			# ALL lines should start with '('
			formula = None
			if elemL[0] in functionSymbols:
				formula = elemL[0]
			elif elemL[0] == 'var':
				varName = elemL[2].replace(')','')
				if elemL[1] == 'bool':
					formula = z3.Bool(varName)
				elif elemL[1] == 'real':
					formula = z3.Real(varName)
				elif elemL[1] == 'int':
					formula = z3.Int(varName)
				else:
					raise Exception('Unknown Variable format: {}'.format(elemL))

			elif elemL[0] == 'const':
				const = elemL[2].replace(')','')
				if elemL[1] == 'real':
					formula = z3.RealVal(const)
				elif elemL[1] == 'int':
					formula = z3.IntVal(const)
				else:
					raise Exception('Unknown Constant format: {}'.format(elemL))
			else:
				logger.error("Unknown sequence : {}\n\t Occured parsing formula: {}".format(elemL[0],string))
				raise Exception

			stack.append(formula)

			closedBrackets = elem.count(')') - 1
			logger.writeToLog("{} - new element in stack: {}\t,cB {}".format(pos ,stack[-1],closedBrackets))
			
			if closedBrackets < 1:
				continue

			while closedBrackets > 0:
				logger.writeToLog('{} - stack: {},{}'.format(pos,stack, closedBrackets))
				tmpPredi = []
				pred = None
				while True:
					pred = stack.pop()
					if isinstance(pred,float) or isinstance(pred,int) or z3.is_int(pred) or z3.is_real(pred) or z3.is_bool(pred):
						tmpPredi.append(pred)
					else:
						break

				if len(tmpPredi) == 1:
					tmpPredi = tmpPredi[0]
					stack.append(mapFunctionSymbol(str(pred))(tmpPredi))
				else:
					tmpPredi = tmpPredi[::-1]
					stack.append(mapFunctionSymbol(str(pred))(tmpPredi))

				logger.writeToLog('{} - {} is applied to {}'.format(pos, pred,tmpPredi))


				logger.writeToLog("{} - new element in stack: {}\t,cB {}".format(pos ,stack[-1],closedBrackets))
				closedBrackets -= 1

			logger.writeToLog('{} - finished :{}:'.format(pos,stack))
	except Exception as e:
		logger.writeToLog("Some Error : {}\n\t Occured parsing formula: {}".format(e,wString))
		raise Exception

	if len(stack) != 1:
		raise Exception("Parsing Error, stack != 1, stack: {}".format(stack))

	# functionBodyAsString = stack[0]
	# weightFunction.set_function_body_as_string(functionBodyAsString)
	logger.writeToLog('finished parsing SMT1 string: {}'.format(stack[0]),'info')

	return stack[0]

def mapFunctionSymbol(symbol):
	if symbol == '&':
		return _and
	elif symbol == '|':
		return _or
	elif symbol == '~':
		return _not
	elif symbol == '<=':
		return _leq
	elif symbol == '<':
		return _le
	elif symbol == '*':
		return _times
	elif symbol == '+':
		return _plus
	elif symbol == '-':
		return _minus
	elif symbol == 'ite':
		return _ite
	else:
		raise Exception("WRONG SYMBOL FOUND : {}".format(symbol))

def recusiveSimplification(formula):
	if formula.children() == 0:
		decl = formula.decl()
		children = []
		for child in formula.children():
			children.append(recusiveSimplification(z3.simplify(child)))
		if len(children) == 1:
			return decl([children[0]])
		else:
			return decl(children)
	else:
		return formula

def _ite(lst):
	if len(lst) != 3 or not z3.is_bool(lst[0]):
		raise Exception('WRONG INPUT FOR _ite: {}'.formula(lst))
	return z3.If(lst[0], lst[1], lst[2])

def _and(lst):
	return z3.And(lst)

def _or(lst):
	return z3.Or(lst)
def _not(lst):
	return z3.Not(lst)
def _le(lst):
	#self._logger.setVerbose(True)
	if len(lst) != 2:
		raise Exception('WRONG INPUT for _le: {}'.format(lst))
	# abstract = abstraction(hybridKnowlegeBase,lst[0] < lst[1],logger)
	return lst[0] < lst[1]

def _leq(lst):
	if len(lst) != 2:
		raise Exception('WRONG INPUT for _le: {}'.format(lst))
	# abstract = abstraction(hybridKnowlegeBase,lst[0] <= lst[1],logger)
	return lst[0] <= lst[1]

def _times(lst):
	if len(lst) != 2:
		# print(lst)
		t = functools.reduce(lambda x, y: x*y, lst)
		# print(t)
		return t
	return lst[0] * lst[1]

def _plus(lst):
	result = 0
	for i in lst:
		result += i
	return result

def _minus(lst):
	result = 0
	for i in lst:
		result +- i
	return result