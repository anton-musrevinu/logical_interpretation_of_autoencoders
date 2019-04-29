#logger
import traceback
import logging
import datetime
from .myExceptions import TestException

class Mylogger(object):

	LEVEL_DEBUG = 0
	LEVEL_INFO = 1
	LEVEL_BENCHMARK = 2
	LEVEL_RESULTS = 3
	LEVEL_TESTING = 4
	LEVEL_ERROR = 9999

	def __init__(self,logger = None, loggerResults = None, name = None, testing = False, level = None):
		self._logger = logger
		self._loggerResults = loggerResults
		self._currentlyTesting = testing
		self._leng = 160
		self._indent = 20
		self._processed = 0
		self._level = Mylogger.LEVEL_INFO if level == None else level

		self._levelOrignial = self._level

		if name != None and not testing:
			self.startBenchMarkProblem(name,phrase = "Starting Benchmark : ", sym = '=')
			self.endBenchMark()

	def set_level(self,level):
		self._level = level
		self._levelOrignial = level

	def get_level(self):
		return self._level

	def setVerbose(self,verbose):
		if verbose:
			self._level = Mylogger.LEVEL_DEBUG
		else:
			self._level = self._levelOrignial

	def getVerbose(self):
		return self._level == Mylogger.LEVEL_DEBUG

	def testing(self):
		return self._currentlyTesting

	def logProcessed(self,nodeId,maxP):
		self._logger.debug('\t Processed: {}/{} -- NodeId: {}'.format(self._processed, nodeId, maxP))
		self._processed += 1

	def resetProgress(self):
		self._processed = 0

	def debug(self,message):
		self.writeToLog(message, level = 'debug')

	def writeToLog(self,message,level = 'debug'):

		# if self._currentlyTesting:
		# 	if self._level == Mylogger.LEVEL_DEBUG or level == 'error':
		# 		print(level + ' --> ' + message)
		# 		return
		# 	elif level == 'info':
		# 		print(level + '--> ' + message)
		# 		return
		# elif self._logger == None:
		# 	return
		# else:

		levels = {}
		levels['debug'] = Mylogger.LEVEL_DEBUG
		levels['info'] = Mylogger.LEVEL_INFO
		levels['benchmark'] = Mylogger.LEVEL_BENCHMARK
		levels['result'] = Mylogger.LEVEL_RESULTS
		levels['error'] = Mylogger.LEVEL_ERROR
		levels['test'] = Mylogger.LEVEL_TESTING

		if level == None:
			raise Exception

		if levels[level] == Mylogger.LEVEL_ERROR:
			if self._currentlyTesting or self._logger == None:
				print(level + '-->' + message)
			else:
				self._logger.error(message)

		#print(message, levels, levels[level], self._level)
		if levels[level] >= self._level:
			if levels[level] == Mylogger.LEVEL_TESTING:
				print(level + '-->' + message)
			elif self._currentlyTesting or self._logger == None:
				print(level + '-->' + message)
			else:
				self._logger.info(level + '-->' + message)

	def result(self,message):
		if self._loggerResults and not self._currentlyTesting:
			self._loggerResults.info(message)
			self.writeToLog(message,'result')
		if self._logger == None and self._loggerResults == None:
			print(message)


	def startBenchMarkProblem(self,name,inputs = None,phrase = 'Starting on BenchMarkProbelm', sym = '-'):

		self.writeToLog(sym * self._leng,'info')
		msg = '{} {}'.format(phrase,name)
		spaces = int((self._leng - (2 * self._indent + len(msg)))/2)
		self.writeToLog(sym * self._indent + ' ' * spaces + msg + ' ' * spaces + sym * self._indent,'info')
		if inputs != None:
			self.benchmarkAddInput(inputs)


	def benchmarkAddInput(self,inputs):
		self.writeToLog("\t" + 'Input: {}'.format(inputs),'info')


	def endBenchMark(self, sym = '='):
		self.writeToLog(sym * self._leng,'info')
		self.writeToLog('\n\n','info')

	def exception(self,exception):
		if self._logger:
			self._logger.exception(exception)
		if self._loggerResults:
			self._loggerResults.exception(exception)

	def error(self,message):
		if self._currentlyTesting:
			print('ERROR: {}'.format(message))
			print(traceback.format_exc())
		elif self._logger:
			self._logger.error(message)
			self._logger.error(traceback.format_exc())
		else:
			print('ERROR:' + message)
			#self._logger.error(traceback.format_exc())

	def testFail(self,message):
		if self._logger:
			self._logger.error(message)
		else:
			print(message)
		raise TestException

	def test(self,message):
		self.writeToLog(message, 'test')

	def startTest(self,testName):
		leng = 180
		indent = 20

		self._currentlyTesting = True

		self.test("-" * leng)
		msg = 'Starting {} tests'.format(testName)
		spaces = int((leng - (2 * indent + len(msg)))/2)
		self.test('-' * indent + ' ' * spaces + msg + ' ' * spaces + '-' * indent)

		self.test("-" * leng)

	def endTest(self):
		leng = 180
		self.test('-' * leng)
		self.test('-' * leng)
		self.test('\n\n')

	def __str__(self):
		return 'logger: self.logger: {}, self._loggerResults: {}, self._verboseAlgorithm: {}, self._currentlyTesting: {}'.format(self._logger,self._loggerResults,self._verboseAlgorithm,self._currentlyTesting)
