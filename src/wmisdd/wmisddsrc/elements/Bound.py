import math
import z3



class Bound(object):

	TYPE_UPPER = 'TYPE_UPPER'
	TYPE_LOWER = 'TYPE_LOWER'

	def __init__(self, typeOfBound, value = None, oldBound = None):
		self._typeOfBound = typeOfBound

		self._floatValue = None
		self._funcList = []

		if value != None:
			self.set(value)

		if oldBound != None:
			self.update(oldBound)

	def has_float(self):
		return self._floatValue != None

	def has_func(self):
		return self._funcList != []

	def get_float_value(self):
		return self._floatValue

	def get_func_list(self):
		return self._funcList

	def set(self, value):
		if value == None or (isinstance(value, float) and abs(value) == math.inf):
			self._floatValue = None
		elif isinstance(value, float) or isinstance(value, int):
			self._floatValue = value
		elif z3.is_rational_value(value) or z3.is_int_value(value):
			self._floatValue = float(str(value))
		else:
			#print(value, type(value))
			self._funcList.append(value)

	def update(self,other):
		if not isinstance(other, Bound):
			raise Exception('Wrong type passed to the Bound.update() method: {}'.format(value))

		if self.has_float() and other.has_float():
			if self._typeOfBound == Bound.TYPE_UPPER:
				self._floatValue = min(self.get_float(), other.get_float())
			else:
				self._floatValue = max(self.get_float(), other.get_float())

		elif not self.has_float() and other.has_float():
			self._floatValue = other.get_float()

		self._funcList.extend(other.get_func_list()[:])

	def get_negated_copy(self):
		if self.is_upper():
			t = Bound(Bound.TYPE_LOWER, oldBound = self)
		else:
			t = Bound(Bound.TYPE_UPPER, oldBound = self)
		return t

	def is_upper(self):
		return self._typeOfBound == Bound.TYPE_UPPER


	def get_float(self):
		if self._floatValue != None:
			return self._floatValue
		elif self._typeOfBound == Bound.TYPE_UPPER:
			return math.inf
		else:
			return -math.inf

	def is_empty(self):
		if self._floatValue == None and not self._funcList:
			return True

	def get_as_python_string(self):
		nbElems = 0
		if (self._floatValue != None and len(self._funcList) > 0) or len(self._funcList) > 1:
			if self._typeOfBound == Bound.TYPE_UPPER:
				result = 'min('
			else:
				result = 'max('
		else:
			result = '('

		if self._floatValue != None or not self._funcList:
			result += str(self.get_float())
			nbElems += 1

		for elem in self._funcList:
			if nbElems == 0:
				result += str(elem)
			else:
				result += ',' + str(elem)

		result += ')'

		return result

	def __str__(self):
		if self.is_upper():
			idx = 'UB:'
		else:
			idx = 'LB:'
		if self.has_float() and self.has_func():
			return '{}({}; {})'.format(idx,self._floatValue, str(self._funcList))
		elif self.has_float():
			return '{}({})'.format(idx,self._floatValue)
		elif self.has_func():
			return '{}({})'.format(idx,str(self._funcList))
		else:
			return '{}()'.format(idx)









