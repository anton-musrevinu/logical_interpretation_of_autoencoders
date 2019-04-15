		

def retriev_all_satisfying_models(sddManager, timeout):

	(indicator, models, execTime) = sddManager.getModelEnumerate(timeout = timeout)
	return models, execTime