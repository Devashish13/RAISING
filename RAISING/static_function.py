
def subset_data(input_data, start, end):
	import pandas
	import numpy
	if isinstance(input_data, pandas.DataFrame):
		return input_data.iloc[:, start:end]
	elif isinstance(input_data, numpy.ndarray):
		return input_data[:, start:end]
	else:
		raise ValueError("Input data must be a pandas DataFrame or a numpy array")

def valid_args(function_cls):
	import inspect
	# Get the signature of the optimizer's constructor
	func_signature = inspect.signature(function_cls)
	# Extract the parameter names from the signature
	param_names = list(func_signature.parameters.keys())
	if 'self' in param_names:
		param_names.remove('self')
	return param_names

def reset_weights(model):
	from keras import backend as K
	from keras.layers import Dense
	import numpy as np
	#session = K.get_session()
	for layer in model.layers:
		if isinstance(layer, Dense):
			old = layer.get_weights()
			if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
				kernel_initializer = layer.kernel_initializer.__class__.from_config(layer.kernel_initializer.get_config())
				bias_initializer = layer.bias_initializer.__class__.from_config(layer.bias_initializer.get_config())
				layer.set_weights([kernel_initializer(layer.kernel.shape),bias_initializer(layer.bias.shape)])

def get_hyperparams(name, param_dict,hp):
	#min_value and max_value should be equal to 0
	if param_dict["min_value"] == 0 and param_dict["max_value"] == 0:
		return 0
	else:
		return hp.Float(name, min_value=param_dict["min_value"], max_value=param_dict["max_value"], sampling=param_dict.get("sampling"))

def random_combnation(param_name, param_dict):
	import numpy as np
	if param_name in ["layers", "units"]:
		return np.random.randint(param_dict["min_value"], param_dict["max_value"] + 1)
	elif "values" in param_dict:  # Categorical variable
		return np.random.choice(param_dict["values"])
	elif param_dict["sampling"] == "log":  #Logarithmic sampling
		return np.random.uniform(np.log10(param_dict["min_value"]), np.log10(param_dict["max_value"]))
	else:  # Linear sampling
		return np.random.uniform(param_dict["min_value"], param_dict["max_value"])

def hpspace_gen(NCombination,hps_dict):
	from RAISING.static_function import random_combnation
	import pandas as pd
	param_combinations = []
	for _ in range(NCombination):
		combination = {}
		for param_name, param_dict in hps_dict.items():
			if "min_value" in param_dict and param_dict["min_value"] == 0 and param_dict["max_value"] == 0:
				continue  # Skip parameters as not included in hp tuning
			combination[param_name] = random_combnation(param_name, param_dict)
		param_combinations.append(combination)
	hpspace = pd.DataFrame(param_combinations)
	return hpspace

def hidden_nodes(shape):
	if shape >= 1000 and shape <= 5000:
		min_node = int(0.01*shape)
		max_node = int(0.05*shape)
	elif shape < 1000:
		min_node = 10
		max_node = 100
	else:
		min_node = 100
		max_node = 300

	return min_node,max_node
