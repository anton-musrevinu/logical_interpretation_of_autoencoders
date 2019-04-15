import math
import z3
from .Function import BoundFunction
from .Bound import Bound

class Interval(object):

    DEF_INTERVAL_FUNCTION = 'DEF_INTERVAL_FUNCTION'
    DEF_INTERVAL_FLOAT = 'DEF_INTERVAL_FLOAT'
    DEF_INTERVAL_ZERO = 'DEF_INTERVAL_ZERO'
    DEF_INTERVAL_DIJUNCT = 'DEF_INTERVAL_DIJUNCT'

    def __init__(self,leadVar,low = None, high = None):
        self._leadVar = leadVar

        self._identifier = None
        self._lowerBound = Bound(Bound.TYPE_LOWER)
        self._upperBound = Bound(Bound.TYPE_UPPER)

        self._lowerBound.set(low)
        self._upperBound.set(high)

        self._update_interval_type()

        self.asString = ''
        self.asStringPrint = ''
        self._func = None
        self._asFunc = None
        self._interval = None

    # def clone_from_other_negated(self,other):
    #     self._lowerBound = other.get_upper_bound()
    #     self._upperBound = other.get_lower_bound()
    #     self._update_interval_type()

    def get_leadVar(self):
        return self._leadVar

    def get_function_string(self):
        return self.asString + '\n#' + self.asStringPrint

    def init_latte_func(self, logger):
        logger.writeToLog('init interval function:','info')

        

    def init_function(self,ii, varsToRefInFunc, logger):

        logger.writeToLog('init interval function: with varsToRefInFunc: {}'.format(varsToRefInFunc),'info')

        if self.is_func():
            self._func = BoundFunction(None, self._lowerBound, self._upperBound,\
                varsToRefInFunc,ii,logger)
            self.asString = self._func.asString
            self.asStringPrint = self._func.asStringPrint
            # self._asFunc = self._func.get()
            #self.subVars = order
        elif self.is_float():
            def limx(*varsToRefInFunc):
                 return [self._lowerBound.get_float(), self._upperBound.get_float()]
            asString = 'def b{}('.format(ii)
            for i, var in enumerate(varsToRefInFunc):
                strvar = str(var)
                if i == 0:
                    asString += '{}'.format(strvar)
                else:
                    asString += ',{}'.format(strvar)
            asString += '):\n\t'
            asStringPrint = asString.replace('def ','').replace('\n\t','')
            # print(asString)
            
            self.asString = asString + 'return [{},{}]'.format(self._lowerBound.get_float(), self._upperBound.get_float())
            self.asStringPrint = asStringPrint + '[{},{}]'.format(self._lowerBound.get_float(), self._upperBound.get_float())
            # self._asFunc = limx

        else:
            raise Exception('function is trying to be created but has status: {}'.format(self._identifier))
            #self.subVars = order

        logger.writeToLog('-finished with func: {}'.format(self.asStringPrint),'info')

        # if not self.min_func and not self.max_func:
        #     def limx(x = None, y = None, z = None, t = None):
        #         return [self.min, self.max]
        # elif not self.min_func:
        #     def limx(x = None, y = None, z = None, t = None):
        #         return [self.min, min(min(self.max_func),self.max)]
        # elif not self.max_func:
        #     def limx(x = None, y = None, z = None, t = None):
        #         return [max(max(self.min_func),self.min), self.max]
        # else:
        #     def limx(x = None, y = None, z = None, t = None):
        #         return [max(max(self.min_func),self.min), min(min(self.max_func),self.max)]

        # return limx

    def as_func(self,logger):
        if self._asFunc == None:
            raise Exception('Function has not been initialized yet')
        else:
            return self._asFunc

    def as_float(self):
        return self._interval

    def _update_interval_type(self):
        intervalAsFloat = self._upperBound.get_float() - self._lowerBound.get_float()
        if intervalAsFloat < 0:
            self._identifier = Interval.DEF_INTERVAL_ZERO
            return
        elif intervalAsFloat == 0:
            self._identifier = Interval.DEF_INTERVAL_ZERO
        else:
            self._identifier = Interval.DEF_INTERVAL_FLOAT
            self._interval = intervalAsFloat

        if self._upperBound.has_func() or self._lowerBound.has_func() and self._identifier != Interval.DEF_INTERVAL_ZERO:
            self._identifier = Interval.DEF_INTERVAL_FUNCTION

    def update_bounds(self,lowerBound, upperBound):
        self._lowerBound.update(lowerBound)
        self._upperBound.update(upperBound)
        self._update_interval_type()


    def is_zero(self):
        return self._identifier == Interval.DEF_INTERVAL_ZERO

    def is_float(self):
        return self._identifier == Interval.DEF_INTERVAL_FLOAT

    def is_func(self):
        return self._identifier == Interval.DEF_INTERVAL_FUNCTION

    def get_upper_bound(self, negate = False):
        if not negate:
            return self._upperBound
        else:
            #print('creating new Bound as Upper: with: {}'.format(self._lowerBound))
            t = Bound(Bound.TYPE_UPPER, self._lowerBound)
            #print('result: {}'.format(t))
            return t

    def get_lower_bound(self,negate = False):
        if not negate:
            return self._lowerBound
        else:
            #print('creating new Bound as Lower: with: {}'.format(self._upperBound))
            t = Bound(Bound.TYPE_LOWER, self._upperBound)
            #print('result: {}'.format(t))
            return t

    def combine_bound(self, newBound, assignment = True):

        if not assignment:
            newBound = newBound.get_negated_copy()

        if self.is_zero() or not self.intersect_bound(newBound):
            self._identifier = Interval.DEF_INTERVAL_ZERO
            return

        if newBound.is_upper():
            self._upperBound.update(newBound)
        else:
            self._lowerBound.update(newBound)
        self._update_interval_type()
        # self.boundVars = list(set(self.boundVars + newBound.))


    def combine(self,other, negate = False):
        if self.is_zero() or other.is_zero():
            self._identifier = Interval.DEF_INTERVAL_ZERO
            self._upperBound = None
            self._lowerBound = None
            return
        if not self.intersect(other, negate):
            self._identifier = Interval.DEF_INTERVAL_ZERO
            self._upperBound = None
            self._lowerBound = None
            return

        self.update_bounds(other.get_lower_bound(negate),other.get_upper_bound(negate))
        self.boundVars = list(set(self.boundVars + other.boundVars))

    def intersect_bound(self,bound):
        if bound.is_upper():
            if bound.get_float() < self._lowerBound.get_float():
                return False
            else:
                return True
        else:
            if self._upperBound.get_float() < bound.get_float():
                return False
            else:
                return True

    def str_with_identifier(self):
        t = ''
        if self._func != None:
            t = self._func.identifier
        return '({}{})'.format(t,str(self))


    def __str__(self):
        if self.asStringPrint != '':
            return self.asStringPrint
        if self.is_zero():
            return '[-]'
        else:
            return '[{},{} - {}]'.format(self._lowerBound, self._upperBound, self._identifier)

    def __eq__(self,other):
        if other == False:
            return False
        if self.get_lower_bound() == other.get_lower_bound()\
         and self.get_upper_bound() == other.get_upper_bound()\
          and self.get_leadVar() == other.get_leadVar():
            return True
        else:
            return False

    def _isNum(self,elem):
        return z3.is_rational_value(elem) or z3.is_int_value(elem)