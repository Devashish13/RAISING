def RSLM(df,hpspace):
	import statsmodels.formula.api as smf
	from kerastuner.engine.hyperparameters import HyperParameters
	from scipy import stats
	import pandas
	import numpy
	#from imblearn.over_sampling import SMOTENC
	# removing constant columns
	const_cols = [col for col in df.columns if df[col].nunique() == 1]
	df.drop(columns=const_cols, inplace=True)
	#q3 = df["score"].quantile(0.75)
	#df = df[df["score"] <= q3]
	min_value = 0.01
	# converting parameters like learning rate and l1, l2 regularization in log10 scale
	filter_cols = [col for col in df.select_dtypes(include=[float, int]).columns if df[col].min() < min_value]
	if "score" in filter_cols:
		filter_cols.remove("score")
	df.loc[:, "score"] = numpy.log10(df.loc[:, "score"])
	for col in filter_cols:
		df.loc[:, col] = numpy.log10(df.loc[:, col])
	input_vars = list(df.columns)
	input_vars.remove("score")
	# fitting a linear model with first order interaction 
	f = "score ~ (" + " + ".join(input_vars) + ")**2"
	lm_model = smf.ols(formula=f, data=df)
	result = lm_model.fit()
	print(result.summary())
	#saving model summary
	with open("RSLM_summary.txt", "w") as file:
		file.write(result.summary().as_text())
	hpspace_df = hpspace
	hpspace_df["score"] = result.predict(hpspace_df[input_vars])
	hpspace_df = hpspace_df.sort_values(by="score")
	best_hp = dict(next(hpspace_df.iloc[[hpspace_df["score"].values.argmin()]].itertuples(index=False))._asdict())
	hpspace_df["score"] = 10**(hpspace_df["score"])
	del best_hp["score"]
	for col in filter_cols:
		best_hp[col] = 10**(best_hp[col])
	print(best_hp)
	for col in filter_cols:
		hpspace_df.loc[:, col] = 10**(hpspace_df.loc[:, col])
	hpspace_df.to_csv("RSLM_prediction.csv",index=False)
	hyperparameters = HyperParameters()
	for key, value in best_hp.items():
		hyperparameters.Fixed(key, value)
	return hyperparameters

