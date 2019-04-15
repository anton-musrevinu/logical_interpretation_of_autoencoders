#Error Hanling

class Error(Exception):
   """Base class for other exceptions"""
   pass

class StopException(Error):
	"""Raising when the Stop Flag is being set"""
	pass

class TestException(Error):
	"""Raising when an test failed"""
	pass

class AlgorithmInitException(Error):
	"""Raising when an alrogithm is intended to be executed by was not initialized first"""
	pass

class AbstractionCondition(Error):
	"""Raising one of the Conditions on the input is not met (found in abtration)"""
	pass

class AbstractionError(Error):
	pass

class TimeoutException(Error):
	pass

class OverFlowException(Error):
	pass

class KnowledgeCompilationError(Error):
	pass

class ModelEnumerationError(Error):
	pass

class RewritingError(Error):
	pass

class IntegrationError(Error):
	pass

class LatteException(IntegrationError):
	pass