import numpy as np
from pandas import DataFrame

def mean(x:np.ndarray) -> np.ndarray:
	"""Calculates the mean of the given array along the axis 0 of the array
	(i.e. column-wise)

	Parameters:
		x [np.ndarray]: np.ndarray containing the differents features along axis 1.
	Return:
		mean [np.ndarray]: array containing the mean of each feature (column)
	"""
	mean = np.nansum(x, axis=0, keepdims=True) / x.shape[0]
	return mean


def std(x:np.ndarray, mean:np.ndarray) -> np.ndarray:
	"""Calculates the standard deviation of each features.

	Parameters:
		x [np.ndarray]: np.ndarray containing the differents features along axis 1.
		mean [np.ndarray]: numpy array containing the mean of each features.
	Return:
		std [np.ndarray]: array containing the standard deviation of each feature (column)
	"""
	m = x.shape[0]
	std = np.sqrt(np.nansum(np.square(x - mean), axis=0, keepdims=True) / m)
	return std


def min(x:np.ndarray) -> np.ndarray:
	""" Extracts the minimum of each features (column of x)
	Parameters:
		x [np.ndarray]: np.ndarray containing the differents features along axis 1.
	Return:
		v_min [np.ndarray]: array (shape[1, m]) containing the minimum of each features.
	"""
	v_min = x[0]
	for j in range(x.shape[1]):
		for i in range(1, x.shape[0]):
			if v_min[j] > x[i,j]:
				v_min[j] = 0
				v_min[j] += x[i,j]
	return v_min.reshape(1,-1)


def max(x:np.ndarray) -> np.ndarray:
	""" Extracts the aximum of each features (column of x)
	Parameters:
		x [np.ndarray]: np.ndarray containing the differents features along axis 1.
	Return:
		v_max [np.ndarray]: array (shape[1, m]) containing the maximum of each features.
	"""
	v_max = x[0]
	for j in range(x.shape[1]):
		for i in range(1, x.shape[0]):
			if v_max[j] < x[i,j]:
				v_max[j] = 0
				v_max[j] += x[i,j]
	return v_max.reshape(1,-1)


def percentile(x:np.array, p:int) -> np.ndarray:
	""" Extracts the percentile p of each features (column of x)
	Parameters:
		x [np.ndarray]: np.ndarray containing the differents features along axis 1.
	Return:
		v_percent [np.ndarray]: array (shape[1, m]) containing the p percentile of each features.
	"""
	index = int(np.round(0.01 * p * x.shape[0]))
	percentiles =np.sort(x, axis = 0)[index]
	return percentiles.reshape(1,-1)


def standardization(df:DataFrame):
	""" Standardized the array according to the formula:
		[x - mean(x)] / [2 * std(x)]
		Only the pandas Series with dtype as numpy.float32 are transformed.
	Parameters:
		* x [pandas.DataFrame]: dataframe to standardized.
		* mean [np.ndarray / float]: vector of mean values (each component is
			the mean of the corresponding columns in the dataframe)
		* std [np.ndarray / float]: vector of std values (each component is	the
			standard deviation of the corresponding columns in the dataframe)
	"""
	if df.empty:
		str_expt = "Exception: dataframe is empty."
		raise Exception(str_expt)
	col_types = df.dtypes.values
	if all([dtype != np.float32 for dtype in col_types]):
		str_expt = "Exception: all series in dataframe are not np.float32 dtype."
		raise Exception(str_expt)
	
	col_names = df.dtypes.index.values
	if not all([ctype == np.float32 for ctype in col_types]):
		str_warning = "Warning: At least one Serie in the DataFrame is not a of type np.float32.\n"
		str_warning += "Series which are not of dtype np.float32 will be ignored."
		print(str_warning)
	
	col_keep = [name for name, ctype in zip(col_names, col_types) if ctype == np.float32]
	v_mean = mean(df[col_keep].values)
	v_std = std(df[col_keep].values, v_mean)
	df.loc[:, col_keep] -= v_mean
	df.loc[:,col_keep] /= (2 * v_std)