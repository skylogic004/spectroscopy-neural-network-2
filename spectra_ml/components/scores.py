import numpy as np
from sklearn import metrics

__author__ = "Matthew Dirks"

def score(df, name, _target, _prediction_column):
	"""
	Args:
		df: DataFrame with columns `_target`, `_prediction_column`, and "set" (which is train, test, or dev)
	"""
	x_groundtruth = df[_target].astype(float)
	y_predictions = df[_prediction_column].astype(float)

	# calculate error/accuracy metrics
	# if (np.any(np.isnan(y_predictions))):
	# 	MAE = MSE = RMSE = np.inf
	# else:
	try:
		MAE = metrics.mean_absolute_error(x_groundtruth, y_predictions)
		MSE = metrics.mean_squared_error(x_groundtruth, y_predictions)
		RMSE = np.sqrt(MSE)

		# use "r^2" to mean pearson correlation squared, which *performs* a linear regression and gets the correlation
		# use "R^2" to mean Coefficient of Determintation, which *only* gets the correlation score, and it equals r^2 iff the input values (y and yhat) are the result of least squares linear regression.
		corr = np.corrcoef(list(x_groundtruth), list(y_predictions))
		r2_pearson = corr[0,1]

		R2_coef_of_det = metrics.r2_score(x_groundtruth, y_predictions)
	except ValueError as e:
		MAE = MSE = RMSE = R2_coef_of_det = np.inf

	return MAE, MSE, RMSE, R2_coef_of_det, len(x_groundtruth)
