
"""
Devashish Tripathi, 12/10/2022

Performing Hyperparameter optimization
"""


def hp_optimization(input_data, output_data,output_class,objective_fun,utag="",Standardize = True,window = None,algorithm="Bayesian",config_file = None,cross_validation = False,model_file = "ANN_architecture_HP.keras",NCombination = 100000,**kwargs):
	"""
	hp_optimization() function implements Bayesian optimization, Hyperband, RandomSearch and RSLM method to perform hyperparameter optimization and return best ANN architecture

	Function implements cross-validation approach along with train-test split.
	"""
	#model_file = "ANN_architecture_HP.keras"

	import os
	import tensorflow
	from tensorflow import keras
	import tensorflow_addons
	import pandas
	from sklearn.model_selection import train_test_split,KFold
	from sklearn.preprocessing import StandardScaler
	import numpy
	import keras_tuner
	import statsmodels.formula.api as smf
	from kerastuner.engine.hyperparameters import HyperParameters
	import inspect
	from imblearn.over_sampling import SMOTENC
	from RAISING.static_function import valid_args,get_hyperparams,hpspace_gen,hidden_nodes
	from RAISING.RSLM_implement import RSLM
	import json

	if output_class not in ["continuous","binary","multiclass"]:
		 raise ValueError(f"Invalid output class:{output_class},\n Only Continuous,Binary and Multiclass outputs are allowed")

#	if algorithm == "RSLM" and hpspace is None:
#		raise ValueError(f"Hyperparameter search space dataframe when RandomSearch_lm hyperparameter tuning algorithm is selected")
	
	if isinstance(output_data,numpy.ndarray):
		output_data = pandas.DataFrame(output_data)

	if config_file is not None:
		with open(config_file, 'r') as file:
			hp_config = json.load(file)


	""" Algo_CVTuner class for performing cross validation with keras tuner hyperparameter tuning methods """

	class Algo_CVTuner(keras_tuner.engine.tuner.Tuner):
		def run_trial(self, trial, x, y,objective,callbacks,args):
			fit_args = args.get('fit_args', {})
			kfold_args = args.get('kfold_args', {})
			cv = KFold(**kfold_args)
			val_losses = []
			losses = []
			for train_indices, test_indices in cv.split(x):
				x_train, x_test = x[train_indices], x[test_indices]
				y_train, y_test = y.iloc[train_indices,:], y.iloc[test_indices,:]
				model = self.hypermodel.build(trial.hyperparameters)
				history = model.fit(x_train, y_train,validation_data = (x_test,y_test),callbacks = callbacks,**fit_args)
				val_losses.append(min(history.history["val_loss"]))
				losses.append(min(history.history["loss"]))

			if isinstance(objective, list):
				self.oracle.update_trial(trial.trial_id, {'multi_objective': numpy.mean(losses) + numpy.mean(val_losses)})
			elif objective == "loss":
				self.oracle.update_trial(trial.trial_id, {"loss": numpy.mean(losses)})
			elif objective == "val_loss":
				self.oracle.update_trial(trial.trial_id, {"val_loss": numpy.mean(val_losses)})

			print(numpy.mean(losses));print(numpy.mean(val_losses))
	
	"""Given the kwargs parameters following *args variables chooses relevant parameters to specific methods. If parameters not given it chooses default values"""

	demo_model = keras.Sequential()
	fit_args = valid_args(function_cls = demo_model.fit)
	fit_args = {key: value for key, value in kwargs.items() if key in fit_args}
	fit_args = {**{'batch_size':32,'epochs':100}, **fit_args}
	print(fit_args)

	compile_args =  valid_args(function_cls = demo_model.compile)
	compile_args = {key: value for key, value in kwargs.items() if key in compile_args}
	if output_class == "continuous":
		compile_args = {**{'loss':"mse",'metrics':[tensorflow_addons.metrics.RSquare()]}, **compile_args}
	elif output_class == "binary":
		compile_args = {**{'loss':"binary_crossentropy",'metrics':["accuracy"]}, **compile_args}
	elif output_class == "multiclass":
		compile_args = {**{'loss':"categorical_crossentropy",'metrics':["accuracy"]}, **compile_args}
	
	print(compile_args)
	
	stop_args = valid_args(function_cls = keras.callbacks.EarlyStopping)
	stop_args = {key: value for key, value in kwargs.items() if key in stop_args}
	stop_args = {**{'patience': 5, 'min_delta': 0.05, 'restore_best_weights': True}, **stop_args}
	print(stop_args)
	
	tuner_class_args = valid_args(function_cls = keras_tuner.Tuner)
	tuner_class_args = {key: value for key, value in kwargs.items() if key in tuner_class_args}
	tuner_class_args = {**{'overwrite': True}, **tuner_class_args}
	print(tuner_class_args)

	dense_args = valid_args(keras.layers.Dense)
	dense_args = {key: value for key, value in kwargs.items() if key in dense_args}
	print(dense_args)

	drop_args = valid_args(keras.layers.Dropout)
	drop_args = {key: value for key, value in kwargs.items() if key in drop_args}
	print(drop_args)

	kfold_args = valid_args(KFold)
	kfold_args = {key: value for key, value in kwargs.items() if key in kfold_args}
	kfold_args = {**{'n_splits':3}, **kfold_args}
	print(kfold_args)

	split_args = valid_args(train_test_split)
	split_args = {key: value for key, value in kwargs.items() if key in split_args}
	split_args = {**{'test_size':0.25}, **split_args}
	print(split_args)

	if algorithm in ["Bayesian","Hyperband","RandomSearch","RSLM"]:
		if cross_validation is False:
			algo_dir = algorithm + "_hpopt" + utag
			output_file = algorithm + "_hpopt_data.csv"
			os.system("mkdir -p "+algo_dir)
		else:
			algo_dir = algorithm + "_hpopt_CV"
			output_file = algorithm + "_hpopt_data_CV.csv"
	else:
		raise ValueError(f"Invalid hyperparameter tuning algorithm choice: {algorithm} \n Only Bayesian optimization, Hyperband, RandomSearch allowed")

	"""Following code subsets input data if a window containing index of subset provided"""

	y_target = output_data

	if window is not None:
#		print("Genotype data loaded")
		start_pos = window[0]
		end_pos = window[1]
		X_data = subset_data(input_data, start_pos, end_pos)
	else:
		X_data = input_data
	print(y_target)
	print(X_data)
	
	scaler = StandardScaler()

	"""Performs Standardization of output data if it is continuous. It uder wants otherwise then can specify Standardize = False"""

	if output_class == "continuous" and Standardize:
		y_target = pandas.DataFrame(scaler.fit_transform(y_target))
	
	X_data = scaler.fit_transform(X_data)

	"""Performs Train Test split on input and output data"""
	
	X_train_full, X_test,y_train_full, y_test = train_test_split(X_data,y_target,**split_args)
	X_train_full_scaled = X_train_full
	X_test_scaled = X_test

	"""
	   build_model_kt() functions utilised by the hyperparameter tuning algorithms to search hyperparameter space.It creates a default hyperparameter space but can be provided 
	   a user defined config file as described in the package
	"""
	def build_model_kt(hp,input_shape=X_data.shape[1:]):
		optimizer_map = {name.lower(): cls for name, cls in vars(keras.optimizers).items() if callable(cls) and name[0].isupper()}
		if config_file:
			hyperparameters = hp_config
			hp_layers = hp.Int('layers', min_value=hyperparameters.get('layers')["min_value"], max_value=hyperparameters.get('layers')["max_value"])
			hp_units = hp.Int('units', min_value=hyperparameters.get('units')["min_value"], max_value=hyperparameters.get('units')["max_value"])
			hp_learning_rate = hp.Float("learning_rate", min_value=hyperparameters.get('learning_rate')["min_value"], max_value=hyperparameters.get('learning_rate')["max_value"], sampling=hyperparameters.get('learning_rate')["sampling"])
			hp_acts = hp.Choice("activation", values=hyperparameters.get('activation')["values"])
			hp_l1 = get_hyperparams("l1_regularizer", hyperparameters.get('l1_regularizer'),hp)
			hp_l2 = get_hyperparams("l2_regularizer", hyperparameters.get('l2_regularizer'),hp)
			hp_drop = get_hyperparams("dropout", hyperparameters.get('dropout'),hp)
			hp_optimizer = hp.Choice("optimizer", values=hyperparameters.get('optimizer')['values'])
			hp_weight_initializer = hp.Choice("weight_initializer", values=hyperparameters.get('weight_initializer')['values'])

		else:
			min_node,max_node = hidden_nodes(shape = X_data.shape[1])
			hp_layers = hp.Int('layers', min_value=1, max_value=3)
			hp_units = hp.Int('units', min_value = min_node, max_value= max_node)
			hp_learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
			hp_acts = hp.Choice("activation",  values = ["relu"])
			hp_l1 = hp.Float("l1_regularizer", min_value=1e-3,max_value=1e-2, sampling="log")
			hp_l2 = hp.Float("l2_regularizer", min_value=1e-5,max_value=1e-4, sampling="log")
			hp_drop = hp.Float("dropout", min_value=0.05,max_value=0.5, sampling="linear")
			hp_optimizer = hp.Choice("optimizer", values=["adam"])
			hp_weight_initializer = hp.Choice("weight_initializer", values=["he_normal"])
		
		model = keras.models.Sequential()
		options = {"input_shape": input_shape}	
			
		for i in range(hp_layers):
			model.add(keras.layers.Dense(hp_units, kernel_initializer=hp_weight_initializer, activation=hp_acts,kernel_regularizer=keras.regularizers.L1L2(l1=hp_l1, l2=hp_l2),**dense_args,**options))
			options = {}
			model.add(keras.layers.Dropout(hp_drop,**drop_args))
		
		if output_class == "continuous":	
			model.add(keras.layers.Dense(y_target.shape[1], **options))
		elif output_class == "binary":
			model.add(keras.layers.Dense(y_target.shape[1],activation="sigmoid", **options))
		else:
			model.add(keras.layers.Dense(y_target.shape[1],activation="softmax", **options))

		optimizer = hp_optimizer
		if optimizer in optimizer_map:
			optimizer_class = optimizer_map[optimizer]
			optimizer_args = valid_args(optimizer_class)
			optimizer_args = {key: value for key, value in kwargs.items() if key in optimizer_args}
			optimizer_u = optimizer_class(learning_rate=hp_learning_rate,**optimizer_args)  # Customize parameters as needed
		else:
			raise ValueError(f"Invalid optimizer choice:{optimizer}")

		model.compile(optimizer=optimizer_u,**compile_args)
		return model

	early_stopping = keras.callbacks.EarlyStopping(**stop_args)

	"""Following code block takes user specified algorithm and performs hyperparameter tuning. Returns and saves the optimum ANN architecture"""

	if algorithm == "Bayesian":
		tuner_args =  valid_args(function_cls = keras_tuner.BayesianOptimization)
		tuner_args = {key: value for key, value in kwargs.items() if key in tuner_args}
		tuner_args = {**{'max_trials':100}, **tuner_args}
		print(tuner_args)
		print(tuner_class_args)
		if cross_validation:
			print(algo_dir)
			args_dict = {'fit_args': fit_args, 'kfold_args': kfold_args}
			tuner = Algo_CVTuner(hypermodel=build_model_kt,oracle=keras_tuner.oracles.BayesianOptimizationOracle(objective = objective_fun,**tuner_args),project_name=algo_dir,**tuner_class_args)
			tuner.search(X_data,y_target,callbacks=[early_stopping],objective = objective_fun,args = args_dict)
		else:
			tuner = keras_tuner.BayesianOptimization(build_model_kt,objective = objective_fun,directory= algo_dir,project_name=algo_dir,**tuner_args,**tuner_class_args)
			tuner.search(X_train_full_scaled,y_train_full, validation_data = (X_test_scaled,y_test),callbacks=[early_stopping],**fit_args)

	elif algorithm == "Hyperband":
		tuner_args =  valid_args(function_cls = keras_tuner.Hyperband)
		tuner_args = {key: value for key, value in kwargs.items() if key in tuner_args}
		tuner_args = {**{'max_epochs':1000}, **tuner_args}
		if cross_validation:
			args_dict = {'fit_args': fit_args, 'kfold_args': kfold_args}
			tuner = Algo_CVTuner(hypermodel=build_model_kt,oracle=keras_tuner.oracles.HyperbandOracle(objective = objective_fun,**tuner_args),project_name=algo_dir,**tuner_class_args)
			tuner.search(X_data,y_target,callbacks=[early_stopping],objective = objective_fun,args = args_dict)
		else:
			tuner = keras_tuner.Hyperband(build_model_kt,objective = objective_fun,directory= algo_dir,project_name=algo_dir,**tuner_args,**tuner_class_args)
			tuner.search(X_train_full_scaled,y_train_full, validation_data = (X_test_scaled,y_test),callbacks=[early_stopping],**fit_args)

	elif algorithm == "RandomSearch" or algorithm == "RSLM":
		tuner_args =  valid_args(function_cls = keras_tuner.RandomSearch)
		tuner_args = {key: value for key, value in kwargs.items() if key in tuner_args}
		tuner_args = {**{'max_trials':1000}, **tuner_args}
		if cross_validation:
			args_dict = {'fit_args': fit_args, 'kfold_args': kfold_args}
			tuner = Algo_CVTuner(hypermodel=build_model_kt,oracle=keras_tuner.oracles.RandomSearchOracle(objective = objective_fun,**tuner_args),project_name=algo_dir,**tuner_class_args)
			tuner.search(X_data,y_target,callbacks=[early_stopping],objective = objective_fun,args = args_dict)
		else:
			tuner = keras_tuner.RandomSearch(build_model_kt,objective = objective_fun,directory= algo_dir,project_name=algo_dir,**tuner_args,**tuner_class_args)
			tuner.search(X_train_full_scaled,y_train_full, validation_data = (X_test_scaled,y_test),callbacks=[early_stopping],**fit_args)
	
	"""
	Following block of code, implements RSLM, performs prediction on user defined hyperparameter space. selects best ANN architecture and save the model.
	Else selects the best ANN architecture from teh other hyperparameter algorithms
	"""

	if algorithm ==  "RSLM":
		if config_file:
			hpspace = hpspace_gen(NCombination,hp_config)
		else:
			min_node,max_node = hidden_nodes(shape = X_data.shape[1])
			default_hps = {"layers": {"min_value": 1, "max_value": 3},
    						"units": {"min_value": min_node, "max_value": max_node},
    						"learning_rate": {"min_value": 1e-4, "max_value": 1e-2, "sampling": "log"},
    						"activation": {"values": ["relu"]},
    						"l1_regularizer": {"min_value": 1e-3, "max_value": 1e-2, "sampling": "log"},
    						"l2_regularizer": {"min_value": 1e-5, "max_value": 1e-4, "sampling": "log"},
    						"dropout": {"min_value": 0.05, "max_value": 0.5, "sampling": "linear"},
    						"optimizer": {"values": ["adam"]},
    						"weight_initializer": {"values": ["he_normal"]}}
			hpspace = hpspace_gen(NCombination,default_hps)
			
		hps = tuner.get_best_hyperparameters(num_trials = tuner_args.get("max_trials"))
		results = []
		for h in hps:
			res = h.values
			results.append(res)
		
		results_df = pandas.DataFrame(results)
		results_df['score'] = 0
		
		best_trials = tuner.oracle.get_best_trials(num_trials=tuner_args.get("max_trials"))
		i = 0
		for trial in best_trials:
			results_df.loc[i,'score'] = trial.score
			i += 1
		
		results_df.to_csv("RandomSearch_trial_result.csv",index = False)
		hyperparameters = RSLM(df = results_df,hpspace = hpspace)
		#model = tuner.hypermodel.build(hyperparameters)
		#model.save(algorithm + "_"+model_file)
		model1 = tuner.hypermodel.build(hyperparameters)
		history = model1.fit(X_train_full_scaled,y_train_full, validation_data = (X_test_scaled,y_test),callbacks=[early_stopping],**fit_args)
		#print(model.summary())
		res_per_epoch = pandas.DataFrame(history.history)
		res_per_epoch.to_csv(output_file,index=False)
		model1.save(model_file)
		return(model1)
	else:
		best_hps = tuner.get_best_hyperparameters()[0]
		print(best_hps.values)
		model1 = tuner.hypermodel.build(best_hps)
		history = model1.fit(X_train_full_scaled,y_train_full, validation_data = (X_test_scaled,y_test),callbacks=[early_stopping],**fit_args)
		res_per_epoch = pandas.DataFrame(history.history)
		res_per_epoch.to_csv(output_file,index=False)
		model1.save(model_file)
		return(model1)

def feature_importance(input_data,output_data,feature_set,model_file="ANN_architecture_HP.keras",iteration = 1,window = None,Standardize = True,feature_method = "DeepFeatImp",train_model_file = "Trained_ANN_Architecture_test.h5", output_class = "continuous",**kwargs):

	"""
	Function for performing the Feature selection using DeepFeatImp, DeepExplainer, KernelExplainer.
	Function requires the ANN architecture from hp_optimization() stage, Input and output data.
	"""
	from numpy.random import seed
	seed(1234)
	import tensorflow
	tensorflow.random.set_seed(3421)

	import os
	from tensorflow import keras
	import pandas
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	import numpy
	import tensorflow_addons
	import shap
	from FeatureImp import DeepFeatImp
	import inspect
	from static_function import subset_data, valid_args,reset_weights 

	if isinstance(output_data,numpy.ndarray):
		output_data = pandas.DataFrame(output_data)
	
	#model1 = model
	demo_model = keras.Sequential()
	        
	"""Given the kwargs parameters following *args variables chooses relevant parameters to specific methods. If parameters not given it chooses default values"""

	fit_args = valid_args(function_cls = demo_model.fit)
	fit_args = {key: value for key, value in kwargs.items() if key in fit_args}
	fit_args = {**{'batch_size':32,'epochs':100}, **fit_args}
	print(fit_args)
	
	stop_args = valid_args(function_cls = keras.callbacks.EarlyStopping)
	stop_args = {key: value for key, value in kwargs.items() if key in stop_args}
	stop_args = {**{'patience': 5, 'min_delta': 0.10, 'restore_best_weights': True}, **stop_args}
	print(stop_args)

	train_test_split_args = valid_args(function_cls = train_test_split)
	train_test_split_args = {key: value for key, value in kwargs.items() if key in train_test_split_args}
	train_test_split_args = {**{'test_size':0.25}, **train_test_split_args}
	print(train_test_split_args)

	"""function single_replicate_feature_imp() performs a single iteration of model training and feature importance. Returns trained model and feature importance dataframe"""

	def single_replicate_feature_imp(model_file,input_data,output_data,start,end,feature_set,window):
		model1 = keras.models.load_model(model_file)
		reset_weights(model1)	
		if window is None:
			data1 = input_data
			features = feature_set
		else:
			data1 = subset_data(input_data,start,end)
			features = feature_set[start:end]
		print(data1.shape)
		scaler = StandardScaler()
		X_data = scaler.fit_transform(data1)

		if output_class == "continuous" and Standardize:
			y_target = pandas.DataFrame(scaler.fit_transform(output_data))
		else:
			 y_target = output_data
		
		X_train, X_test,y_train, y_test = train_test_split(X_data,y_target,**train_test_split_args)

		if feature_method == "DeepExplainer":
			shap_args = valid_args(function_cls = shap.DeepExplainer.shap_values)
			shap_args = {key: value for key, value in kwargs.items() if key in shap_args}
			print(shap_args)

			model1.fit(X_train,y_train,validation_data = (X_test,y_test),callbacks = [keras.callbacks.EarlyStopping(**stop_args)],**fit_args)
			e = shap.DeepExplainer(model1, X_test)
			feature_imp = e.shap_values(X_test,**shap_args)
			print(feature_imp[0].mean(axis = 0).shape)
			shap_mean = [numpy.abs(feature_imp[i]).mean(axis = 0) for i in range(output_data.shape[1])]
			shap_mean.insert(0, features)
			importance_df = pandas.DataFrame(shap_mean).T
			out_cols = output_data.columns.to_list()
			out_cols.insert(0,"features")
			importance_df.columns = out_cols

		elif feature_method == "KernelExplainer":
			shap_args = valid_args(function_cls = shap.KernelExplainer.shap_values)
			shap_args = {key: value for key, value in kwargs.items() if key in shap_args}
			print(shap_args)

			shap_sample_args = valid_args(function_cls = shap.sample)
			shap_sample_args = {key: value for key, value in kwargs.items() if key in shap_sample_args}
			print(shap_sample_args)

			model1.fit(X_train,y_train,validation_data = (X_test,y_test),callbacks = [keras.callbacks.EarlyStopping(**stop_args)],**fit_args)
			X_test1 = shap.sample(X_test, **shap_sample_args)
			e = shap.KernelExplainer(model1, X_test1)
			feature_imp = e.shap_values(X_test1,**shap_args)
			shap_mean = [numpy.abs(feature_imp[i]).mean(axis = 0) for i in range(output_data.shape[1])]
			shap_mean.insert(0, features)
			importance_df = pandas.DataFrame(shap_mean).T
			out_cols = output_data.columns.to_list()
			out_cols.insert(0,"features")
			importance_df.columns = out_cols
		elif feature_method == "DeepFeatImp":
			print("Generalized Feature Importance")
			model1.fit(X_data,y_target,callbacks = [keras.callbacks.EarlyStopping(monitor='loss',**stop_args)],**fit_args)
			importance_df = DeepFeatImp(model1,feature_list = features,output_list = output_data.columns.to_list())
		else:
			 raise ValueError(f"Invalid Feature Importance algorithm choice: {algorithm} \n DeepExplainer, KernelExplainer, GenDeepFeatImp allowed")

		return [importance_df,model1]

	"""function to call single_replicate_feature_imp inside mean_iterate() function"""
	def single_iterate(_):
		df = single_replicate_feature_imp(model_file,input_data,output_data,start,end,feature_set,window)
#               print(df)
		return(df)

	""" 
	function implements single_replicate_feature_imp() iteration(N) times and estimates mean and biases of N models and sets the means to get final model.
	The final feature importance is mean of feature importance across N replicates.
        """
	def mean_iterate(iteration):
		iter_result = list(map(single_iterate,range(iteration)))
		data = numpy.array([df[0].values for df in iter_result])
		data_without_id = data[:, :, 1:]
		column_means = numpy.mean(data_without_id, axis=0)
		mean_df = pandas.DataFrame(column_means, columns=iter_result[0][0].columns[1:])
		mean_df.insert(0, 'features', iter_result[0][0]['features'])
		
		model_res = [df[1] for df in iter_result]
		model_layers = [layer for layer in model_res[0].layers if isinstance(layer, keras.layers.Dense)]
		weights = [layer.get_weights()[0] for layer in model_layers]
		biases = [layer.get_weights()[1] for layer in model_layers]
		print(len(model_res))
		for i in range(1,len(model_res)):
			model = model_res[i]
			model_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
			weights1 = [layer.get_weights()[0] for layer in model_layers]
			biases1 = [layer.get_weights()[1] for layer in model_layers]
			for i in range(len(model_layers)):
				weights[i] = weights[i] + weights1[i]
				biases[i] = biases[i] + biases1[i]
		
		for i in range(len(model_layers)):
			weights[i] = weights[i]/iteration
			biases[i] = biases[i]/iteration

		for i in range(len(model_layers)):
			layer = model_layers[i]
			layer.set_weights([weights[i],biases[i]])

		return([mean_df,model_res[0]])

	if window is None :
		start = 0
		end = input_data.shape[1]
		print(end)
		res_model_featimp = mean_iterate(iteration)
		findf =  res_model_featimp[0]
		model_fin = res_model_featimp[1]
		model_fin.save(train_model_file)
		findf.to_csv("feature_importance.csv",index=False)
		print(findf)
		return(findf)
	else:
		start = window[0]
		end = window[1]
		res_model_featimp = mean_iterate(iteration)
		findf =  res_model_featimp[0]
		model_fin = res_model_featimp[1]
		model_fin.save(train_model_file)
		findf.to_csv(f"feature_importance_{start}_{end}.csv",index=False)
		print(findf)
		return(findf)
