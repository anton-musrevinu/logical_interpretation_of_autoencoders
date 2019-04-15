import types
import sys, imp,ast
import importlib.util
from importlib import import_module
import z3

class Function:

	def __init__(self, tmpFile, order = None):
		self._variables = []
		self._foo = None
		self._tmpFile = tmpFile
		self._variableOrder = {}
		if order:
			for i, var in enumerate(order):
				self._variableOrder[i] = var

		self._variableOrderBool = {}

	def _parseWriteAndRead(self,fucntionAsString):
		raise Exception('Not yet implemented')

	def __str__(self):
		return self._asString

	def getVariableOrder(self):
		return self._variableOrder

	def getVariableOrderBool(self):
		return self._variableOrderBool

	def eval(self):
		return self._foo(variables)

	def get(self):
		return self._foo

class BoundFunction(Function):
	def __init__(self,tmpFile, lowerBound, upperBound,order, name, logger):
		self._logger = logger
		Function.__init__(self,tmpFile, order)
		#self._logger.setVerbose(False)

		functionAsPythonString = self._parseListsToPythonCode(name, lowerBound, upperBound)
		self.asString = functionAsPythonString
		# functionAsPythonString = 'from math import inf\n\n' + functionAsPythonString
		# self._parseWriteAndRead(functionAsPythonString)


	def _parseWriteAndRead(self,functionAsPythonString):
		with open(self._tmpFile,'w') as f:
			for line in functionAsPythonString:
				f.write(line)

		spec = importlib.util.spec_from_file_location("functions", self._tmpFile)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		self._foo = module.boundFunction
		return

	def _parseListsToPythonCode(self,name, lowerBound, upperBound):

		body_str = '[' + lowerBound.get_as_python_string() + ',' + upperBound.get_as_python_string() + ']'
		body = '\treturn {}\n'.format(body_str)
		# self._logger.writeToLog('Body of the Function: {}'.format(body))

		head = 'def b{}('.format(name)
		for i, var in self._variableOrder.items():
			strvar = str(var)
			if i == 0:
				head += '{}'.format(strvar)
			else:
				head += ',{}'.format(strvar)
		head += '):\n'
		self.identifier = head.replace('def ','').replace('\n','')
		self.asStringPrint = self.identifier + body.split('return')[1].replace('\n','')
		# self._logger.writeToLog('Init of the function: {}'.format(head))

		fullFuncAsString = head + body
		# self.asString = body.split('return')[1].replace('\n','')
		self._logger.writeToLog('Full Function: \n\n{}'.format(fullFuncAsString))
		# self._logger.writeToLog('Full Var Order: {}'.format(self._variableOrder.values()))
		return fullFuncAsString

# class WeightFunctions(object):
# 	def __init__(self, name, func_list, tmpDir, logger, order = None):
# 		self._name = name
# 		self.func_list = func_list
# 		self._tmpFile = tmpDir + self._name + '_functions.py'
# 		self._logger = logger
# 		self._variableOrderBool = []


# 	def construct_function_with_order(self,order, func_idxs):
# 		self._variableOrder = order

# 		body = '\treturn ' + func_idxs[0]
# 		for i in func_idxs[1:]:
# 			body = '*' + func_list[i]
# 		body = body + '\n'

# 		self._logger.writeToLog('Body of the Function: {}'.format(body))

# 		for i, var in self._variableOrder.items():
# 			strvar = str(var)
# 			if i == 0:
# 				head = 'def trueWeightFunc({}'.format(strvar)
# 			else:
# 				head += ',{}'.format(strvar)
# 		for i, var in self._variableOrderBool.items():
# 			head += ',{}'.format(var)
# 		head += '):\n'
# 		self._identifier = head.replace('def trueWeightFunc','foo').replace('\n','')
# 		self._logger.writeToLog('Init of the function: {}'.format(head))

# 		fullFuncAsString = head + body
# 		self._logger.writeToLog('Full Function: \n\n{}'.format(fullFuncAsString))
# 		self._logger.writeToLog('Full Var Order: {}'.format(self._variableOrder.values()))
# 		return fullFuncAsString

class WeightFunction:

	DEF_VAR_ORDER_ALPH = 'DEF_VAR_ORDER_ALPH'
	DEF_VAR_ORDER_NBREF = 'DEF_VAR_ORDER_NBREF'

	def __init__(self):
		self._boundRealVariables = []
		self._boundBoolVariables = []
		self._foo = None
		self._wfAsZ3 = None
		self._wfBodyAsString = None

		self._single = True
		self._identifier = ''

		self._orderedRealVariables = None
		self._orderedBoolVariables = None

	def is_one(self):
		return self._wfBodyAsString == '1'

	def contains_variable(self, var):
		if isinstance(var,z3.ArithRef):
			return var in self._boundRealVariables
		elif isinstance(var,z3.BoolRef):
			return var in self._boundBoolVariables

	def add_variable(self,var):
		if isinstance(var,z3.ArithRef):
			self._boundRealVariables.append(var)
		elif isinstance(var,z3.BoolRef):
			self._boundBoolVariables.append(var)

	def get_bound_variables(self):
		return self._boundRealVariables, self._boundBoolVariables

	def set_variable_order(self, orderReal, orderBool):
		self._orderedRealVariables = orderReal
		self._orderedBoolVariables = orderBool

	def set_function_as_z3(self,function):
		self._wfAsZ3 = function

	def get_function_as_z3(self):
		return self._wfAsZ3

	def set_function_body_as_string(self, wfBodyAsString):
		self._wfBodyAsString = wfBodyAsString

	def set_func_and_identifier(self,foo, identifier):
		self._foo = foo
		self._identifier = identifier

	def get_function_body_as_string(self):
		return self._wfBodyAsString

	def is_single(self):
		return self._single

	def get_real_variable_order(self):
		return self._orderedRealVariables

	def get_bool_variable_order(self):
		return self._orderedBoolVariables

	def get_identifiert(self):
		return self._identifier

	def get(self):
		if self._foo == None:
			raise Exception('WF has not been initialized yet: ' + str(self))
		return self._foo

	def __str__(self):
		return 'WFO: id: {}, realOrder: {}, boolOrder: {}, init: {}, single: {}'.format(\
			self._identifier, self._orderedRealVariables, self._orderedBoolVariables, self._foo != None, self._single)