

class WmiResult(object):

	INDICATOR_OVERFLOW = -1
	INDICATOR_KC_TIMEOUT = -2.2
	INDICATOR_KC_UNKOWN  = -2.1
	INDICATOR_ME_TIMEOUT = -3.2
	INDICATOR_ME_OVERFLOW = -3.1
	INDICATOR_INT_TIMEOUT = -4.2
	INDICATOR_UNKNOWN = -3
	INDICATOR_ALLGOOD = 0

	def __init__(self, name):
		self._name = name
		self._indicator = WmiResult.INDICATOR_ALLGOOD

		self._abstractionTime = 0
		self._sddCreationTime = 0

		self._sddQueryTime = 0
		self._modelCount = -1

		self._wmiTime = 0
		self._findConditionTime = 0
		self._constructTime = 0

		self._wmiResult = None
		self._totaltime = -1

	def set_abstraction_time(self,time):
		self._abstractionTime = time

	def set_sdd_creation_time(self,time):
		self._sddCreationTime = time

	def set_sdd_query_time(self,time):
		self._sddQueryTime = time

	def set_model_count(self,modelcount):
		self._modelCount = modelcount

	def get_sdd_query_time(self):
		return self._sddQueryTime

	def get_model_count(self):
		return self._modelCount

	def set_wmi_int_time(self,time):
		self._wmiTime = time

	def set_rewrite_time(self,time):
		self._rewriteTime = time

	def set_construction_time(self,time):
		self._constructTime = time

	def set_condition_time(self,time):
		self._rewriteTime = self._rewriteTime + time

	def set_total_time(self,time):
		self._totaltime = time

	def set_wmi_result(self, wmiValue, time, integrationTime, nbIntegrations):
		self._wmiResult = wmiValue
		self._wmiTime = time
		self._IntegrationTime = integrationTime
		self._nbIntegrations = nbIntegrations

	def get_total_time(self):
		if self._totaltime < 0:
			return self._abstractionTime + self._rewriteTime + self._sddCreationTime + self._sddQueryTime + self._constructTime + self._wmiTime
		else:
			return self._totaltime

	def get_result(self):
		return self._wmiResult

	def get_times_tuple(self):
		absTuple = self._create_time_tuble(self._abstractionTime)
		rewriteTuple = self._create_time_tuble(self._rewriteTime)
		creatTuple = self._create_time_tuble(self._sddCreationTime)
		queryTuple = self._create_time_tuble(self._sddQueryTime)
		constructTime = self._create_time_tuble(self._constructTime)
		intTuple = self._create_time_tuble(self._wmiTime)
		return absTuple, rewriteTuple, creatTuple, queryTuple, constructTime, intTuple

	def get_times_string(self):
		try:
			string = 'TotalTime: {:.4}, 1.A: {},2:RP: {}, 3.KC: {}, 4.ME: {}, 5.CS: {}, 6.I: {}'.format(self.get_total_time(), *self.get_times_tuple())
		except ValueError:
			string = 'TotalTime: {}, 1.A: {},2:RP: {}, 3.KC: {}, 4.ME: {},6.CS: {} , 6.I: {}'.format(self.get_total_time(), *self.get_times_tuple())

		return string

	def _create_time_tuble(self,instTime):
		try:
			timeTuple = ('{:.2}s'.format(instTime), '{:.2}%'.format(instTime/self.get_total_time()))
		except ValueError:
			timeTuple = ('{}s'.format(instTime), '{}%'.format(instTime/self.get_total_time()))
		return timeTuple

	def is_all_good(self):
		return self._indicator == WmiResult.INDICATOR_ALLGOOD

	def set_indicator(self,indicator):
		self._indicator = indicator

	def __str__(self):
		if self._indicator == WmiResult.INDICATOR_ALLGOOD:
			return 'Value: {}, {}, nbInt: {}'.format(self.get_result(), self.get_times_string(), self._nbIntegrations)
		else:
			return 'ERROR: {}, {}'.format(self._indicator, self.get_times_string())






class QueryResult:
	def __init__(self,name):
		self._name = name
		self._base_wmi_result = None
		self._query_wmi_result = None

	def add_base_wmi_result(self, wmiResult):
		self._base_wmi_result = wmiResult

	def add_query_wmi_result(self, wmiResult):
		self._query_wmi_result = wmiResult

		self._prob = self._query_wmi_result.get_result()/self._base_wmi_result.get_result()

	def check_resulting_prob(self):
		if self._prob < 0:
			raise Exception('resulting probability: {} is smaller than 0'.format(self._prob))
		if self._prob > 1:
			raise Exception('resulting probability: {} is largen than 1'.format(self._prob))

	def get_prob(self):
		return self._prob

	def get_base_wmi(self):
		return self._base_wmi_result

	def get_total_time(self):
		if self._query_wmi_result != None:
			return self._base_wmi_result.get_total_time() + self._query_wmi_result.get_total_time()
		else:
			return self._base_wmi_result.get_total_time()

	def get_times(self):
		if self._query_wmi_result != None:
			return (self._base_wmi_result.get_times(), self._query_wmi_result.get_times())
		else:
			return self._base_wmi_result.get_times()
