from pandas import read_csv, concat, DataFrame
from os.path import exists, isfile
from numpy import ndarray, array, float64
import json
from sys import exit

# =========================================================================== #
#                        | Definition des constantes|                         #
# =========================================================================== #
from constants import expected_col, default_train_file, default_predict_file, \
	nb_features, nb_classes, dct_types, other_cols
END = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
BLACK = '\033[1;30m'
RED = '\033[1;31m'
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[1;34m'
VIOLET = '\033[1;35m'
CYAN = '\033[1;36m'
WHITE = '\033[1;37m'

# --------------------------------------------------------------------------- #
# Functions to display the usages                                             #
# --------------------------------------------------------------------------- #
def print_predict_usage():
	""" Prints the usage for train program.
	"""
	str_usage = BLUE + "Usage:\n" + END
	str_usage += "  python logreg_predict.py --dataset=... --coefficients=...\n"
	str_usage += BLUE + "Args:\n" + END
	str_usage += YELLOW + "  --dataset=[csv file]:\n" + END
	str_usage += "     * dataset on which the prediction will be performed.\n"
	str_usage += f"       (default) path is set to {default_predict_file}.\n"
	str_usage += YELLOW + "  --coefficients=[json file]:\n" + END
	str_usage += "     * coefficients used to retrieve the model after training.\n"
	str_usage += f"       (default) None, without correct JSON file, nothing is done.\n"
	print(str_usage)

def print_train_usage():
	""" Prints the usage for train program.
	"""
	str_usage = BLUE + "Usage:\n" + END
	str_usage += "  python logreg_train.py --dataset=... --graphic=... --method=...\n"
	str_usage += BLUE + "Args:\n" + END
	str_usage += YELLOW + "  --graphic=[console/static]\n" + END
	str_usage += "     * console: (default) no graphic, only results in terminal.\n"
	str_usage += "     * static: display 2 plots: the raw data with the model curve\n"
	str_usage += "             and the cost function with respect to the iteration.\n"
	str_usage += YELLOW + "  --method=[GD/SGD/SGD+momentum/minibatch]:\n" + END
	str_usage += "     * GD: (default) Gradient descent method.\n"
	str_usage += "     * SGD: Stochastic Gradient Descent method.\n"
	str_usage += "     * SGD: Stochastic Gradient-Descent with momentum.\n"
	str_usage += "     * minibatch: mini-batch method.\n"
	str_usage += YELLOW + "  --dataset=[csv file]:\n" + END
	str_usage += "     * dataset on which the training will be performed.\n"
	str_usage += f"       (default) path is set to {default_train_file}.\n"
	print(str_usage)

# --------------------------------------------------------------------------- #
# Functions to display the usages                                             #
# --------------------------------------------------------------------------- #
def open_read_file(filename:str, context:str):
	""" Reads the data from the data file filename.
	Parameters:
		choice [str]: ("all","train","test", ...) which data to include in
			the dataframe.
	Return:
		df [pandas.DataFrame]: dataframe containing the data.
		None: If file does not exist or is corrupted.
	"""
	_verbose = False
	if _verbose:
		print("Reading the dataset.")
	
	if not exists(filename):
		str_err = f"{filename} does not exist"
		print(RED + str_err + END)
		return None
	try:
		df = read_csv(filename, sep=',', dtype=dct_types)
	except:
		str_err = f"{filename} not a CSV file or is corrupted or totally empty."
		print(RED + str_err + END)
		return None
	
	name_cols = list(df.columns)
	df_type = df.dtypes

	if not all([name in expected_col for name in name_cols]):
		str_err = "Dataset contains at least an unexpected column " + \
			"or no header is specified in CSV or an excpected column is missing."
		print(RED + str_err + END)
		return None
	
	if df.shape[1] != 19:
		print(RED + "Dataset is expected to have 19 columns." + END)
		return None

	if df.shape[0] == 0:
		print(RED + "Dataset seems to be empty." + END)
		return None
	
	if context == "train":
		other_cols.remove("Hogwarts House")
		df.drop(axis=1, labels=other_cols, inplace=True)
	if context == "predict":
		df.drop(axis=1, labels=other_cols, inplace=True)
	return df


# --------------------------------------------------------------------------- #
# Function for read and write models into a json                              #
# --------------------------------------------------------------------------- #
def check_shape_list(dct:dict, key:str) -> bool:
	"""
	... Docstring ...
	"""
	lst = dct[key]
	if key == "weights":
		if len(lst) != nb_classes:
			return False
		for l_elem in lst:
			if len(l_elem) != nb_features:
				return False
	if key == "biais":
		if len(lst) != nb_classes:
			return False
		for l_elem in lst:
			if len(l_elem) != 1:
				return False
	return True


def open_read_coeff(modelpath:str, n_features:int, n_classes:int) -> ndarray:
	""" Reads if possible the JSON file containing the model/hypothesis's
	coefficients. If the file does not exist or is not a correct JSON file
	theta vector is set to numpy.ndarray[([0.0], [0.0]]).
	Return:
		theta: (np.ndarray) hypothesis's coefficients vector
	"""
	if not isfile(modelpath):
		print(f'{modelpath} does not exist.')
		return None
	else:
		with open(modelpath, 'r') as open_file:
			try:
				model = json.load(open_file)
			except:
				print(f'{modelpath} is either empty, corrupted or not a json file.')
				return None

		# -- Checking the compisition (types of element) of the model
		# -- retrieved from the json file.
		if any([key not in ["weights", "biais"] for key in model.keys()]):
			print("Unexpected key(s) for model.")
			return None
		if any([key not in model.keys() for key in ["weights", "biais"]]):
			print("Missing key(s) for model.")
			return None
		if not isinstance(model["weights"], list):
			print(f'{modelpath} seems to be corrupted.')
			return None
		if not isinstance(model["biais"], list):
			print(f'{modelpath} seems to be corrupted.')
			return None
		if any([not isinstance(m_w, list) for m_w in model["weights"]]):
			print(f'{modelpath} seems to be corrupted.')
			return None
		if any([not isinstance(m_b, list) for m_b in model["biais"]]):
			print(f'{modelpath} seems to be corrupted.')
			return None
		m_w = sum(model["weights"], [])
		m_b = sum(model["biais"], [])
		if any([not isinstance(w, (float, int)) for w in m_w]):
			print(f'{modelpath} seems to be corrupted.')
			return None
		if any([not isinstance(b, (float, int)) for b in m_b]):
			print(f'{modelpath} seems to be corrupted.')
			return None
		if not check_shape_list(model, "weights") \
			or not check_shape_list(model, "biais"):
			print(f'{modelpath} has not the good shape.')
			return None

		model["weights"] = array(model["weights"])
		model["biais"] = array(model["biais"])
		if (model["weights"].shape != (n_classes,n_features)) or (model["biais"].shape != (n_classes,1)):
			print("Unexpected shape for models.")
			return None
		else:
			print("All keys in models.json are valid.")
	for i in range(n_classes):
		model[f"weights_{i+1}"] = model["weights"][i,:].reshape(-1,1)
		model[f"biais_{i+1}"] = model["biais"][i][0]
	return model


def open_write_coeff(weights:ndarray, b:ndarray):
	""" Writes if possible the theta in JSON file containing.
	If the file does not exist then it is created.
	"""
	serialized_weights = weights.tolist()
	serialized_biais = b.tolist()
	parameters = {"weights":serialized_weights, "biais":serialized_biais}

	if not isfile('model.json'):
		print('Creation of JSON file model.json.')
	with open("model.json", 'w') as open_file:
		try:
			json.dump(parameters, open_file, indent=4)
			print("Weights and biais components have been written in model.json.")
		except:
			print('Error during writting of coefficient in JSON file.')