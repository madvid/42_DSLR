
from pandas import read_csv, concat, DataFrame
from os.path import exists, isfile
from numpy import ndarray, array, float64
import json

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

expected_col = ["Index", "Hogwarts House", "First Name", "Last Name",
				"Birthday", "Best Hand", "Arithmancy", "Astronomy",
				"Herbology", "Defense Against the Dark Arts", "Divination",
				"Muggle Studies", "Ancient Runes", "History of Magic",
				"Transfiguration", "Potions", "Care of Magical Creatures",
				"Charms", "Flying"]

# --------------------------------------------------------------------------- #
# Functions to display the usages                                             #
# --------------------------------------------------------------------------- #
def print_predict_usage():
	""" Prints the usage for predict program.
	"""
	str_usage = BLUE + "Usage:\n" + END
	str_usage += "  python logreg_predict.py --graphic=... --data=...\n"
	str_usage += BLUE + "Args:\n" + END
	str_usage += YELLOW + "  --graphic=[console/static/dynamic]\n" + END
	str_usage += "     * console: (default) no graphic, only results in terminal.\n"
	str_usage += "     * static: display 2 plots: the raw data with the model curve\n"
	str_usage += "             and the accuracy, recall and F1 scores respect to the iteration.\n"
	str_usage += "     * dynamic: diplay 3 plots: data with model curve.\n"
	str_usage += "                plus the cost function with respect to the iteration.\n"
	str_usage += "                plus the contour plot of cost function and current cost value (red dot).\n"
	str_usage += YELLOW + "  --data=[csv file]:" + END
	str_usage += "     * data on which the prediction will be done and evaluated.\n"
	print(usage)


def print_train_usage():
	""" Prints the usage for train program.
	"""
	str_usage = BLUE + "Usage:\n" + END
	str_usage += "  python logreg_train.py --graphic=... --method=...\n"
	str_usage += BLUE + "Args:\n" + END
	str_usage += YELLOW + "  --graphic=[console/static/dynamic]\n" + END
	str_usage += "     * console: (default) no graphic, only results in terminal.\n"
	str_usage += "     * static: display 2 plots: the raw data with the model curve\n"
	str_usage += "             and the cost function with respect to the iteration.\n"
	str_usage += "     * dynamic: diplay 3 plots: data with model curve.\n"
	str_usage += "                plus the cost function with respect to the iteration.\n"
	str_usage += "                plus the contour plot of cost function and current cost value (red dot).\n"
	str_usage += YELLOW + "  --method=[GD/SGD/minibatch]:\n" + END
	str_usage += "     * GD: Gradient-Descent: (default) gradient descent method.\n"
	str_usage += "     * SGD: Stochastic Gradient-Descent: stochastic gradient descent method.\n"
	str_usage += "     * minibatch: mini-batch method.\n"
	str_usage += YELLOW + "  --data=[csv file]:\n" + END
	str_usage += "     * dataset on which the training will be performed.\n"
	print(str_usage)


# --------------------------------------------------------------------------- #
# Functions to display the usages                                             #
# --------------------------------------------------------------------------- #
def open_read_file(choice:str="all"):
	""" Reads the data from the data file filename.
	Parameters:
		choice [str]: ("all","train","test") which data to include in the dataframe.
	Return:
		df [pandas.DataFrame]: dataframe containing the data.
		None: If file does not exist or is corrupted.
	"""
	_verbose = False
	if _verbose:
		print("Reading the dataset.")
	if choice == "dataset_train.csv":
		if not exists("datasets/dataset_train.csv"):
			str_err = "dataset/dataset_train.csv does not exist"
			print(str_err)
			return None
		try:
			df = read_csv("datasets/dataset_train.csv")
		except:
			str_err = "dataset_train.csv not a CSV file or is corrupted."
			print(str_err)
			return None
	if choice == "dataset_test.csv":
		if not exists("datasets/dataset_test.csv"):
			str_err = "dataset/dataset_test.csv does not exist"
			print(str_err)
			return None
		try:
			df = read_csv("datasets/dataset_test.csv")
		except:
			str_err = "dataset_test.csv not a CSV file or is corrupted."
			print(str_err)
			return None
	if choice == "all":
		if exists("datasets/dataset_train.csv") and exists("datasets/dataset_test.csv"):
			str_err = "dataset/dataset_train.csv or/and" \
				+ "does/do not exist."
			print(str_err)
			return None
		try:
			df1 = read_csv("datasets/dataset_train.csv", index_col=True)
			df2 = read_csv("datasets/dataset_test.csv", index_col=True)
		except:
			str_err = "dataset_train.csv and / or dataset_test.csv is/are" \
				+ " not a CSV file or is/are corrupted."
			print(str_err)
			return None
		df = concat([df1, df2], axis=0, ignore_index=True)
	
	name_cols = df.columns.values
	df_type = df.dtypes

	if not (all([name in expected_col for name in name_cols]) \
		and ([col in name_cols for col in expected_col])):
		print(RED + "Dataset does not contain exactly the expected columns." + END)
		return None
	
	if df.shape[1] != 19:
		print(RED + "Dataset is expected to have 19 columns." + END)
		return None

	if df.shape[0] == 0:
		print(RED + "Dataset seems to be empty." + END)
		return None
	
	numerical_col = df_type[(df_type == "float64") | (df_type == "int64")].index.values
	other_cols = [elem for elem in name_cols if elem not in numerical_col]
	df.drop(axis=1, labels=other_cols, inplace=True)
	df.drop(axis=1, labels="Index", inplace=True)
	return df


# --------------------------------------------------------------------------- #
# Function for read and write models into a json                              #
# --------------------------------------------------------------------------- #
def open_read_coeff(n_features:int, n_classes:int) -> ndarray:
	""" Reads if possible the JSON file containing the model/hypothesis's
	coefficients. If the file does not exist or is not a correct JSON file
	theta vector is set to numpy.ndarray[([0.0], [0.0]]).
	Return:
		theta: (np.ndarray) hypothesis's coefficients vector
	"""
	if not isfile('models.json'):
		print('JSON file models.json does not exist.')
		print('model is initiated to None')
		return None
	else:
		with open("models.json", 'r') as open_file:
			try:
				models = json.load(open_file)
			except:
				print('JSON file models.json seems to be corrupted.')
				print('models is initiated to None')
				return None

		if any([key not in ["weights", "biais"] for key in models.keys()]):
			print("Unexpected key(s) for models.")
			print('models is initiated to None')
			return None
		models["weights"] = array(models["weights"])
		models["biais"] = array(models["biais"])
		if (models["weights"].shape != (n_classes,n_features)) or (models["biais"].shape != (n_classes,1)):
			print("Unexpected shape for models.")
			print('models is initiated to None')
			return None
		else:
			print("All keys in models.json are valid.")
	for i in range(n_classes):
		models[f"weights_{i+1}"] = models["weights"][i,:].reshape(-1,1)
		models[f"biais_{i+1}"] = models["biais"][i][0]
	return models


def open_write_coeff(weights:ndarray, b:ndarray):
	""" Writes if possible the theta in JSON file containing.
	If the file does not exist then it is created.
	"""
	serialized_weights = weights.tolist()
	serialized_biais = b.tolist()
	
	parameters = {"weights":serialized_weights, "biais":serialized_biais}
	if not isfile('coefficients.json'):
		print('Creation of JSON file coefficients.json.')
	with open("models.json", 'w') as open_file:
		try:
			json.dump(parameters, open_file, indent=4)
			print("Weights and biais components have been written in coefficients.json.")
		except:
			print('Error during writting of theta in JSON file.')


def open_read_data(filepath:str, **kwargs) -> DataFrame:
	""" Reads if possible the CSV file containing the data.
	If the file does not exist or is not a correct CSV file
	the program quit.
	Return:
		df: (pd.DataFrame) dataframe containing the raw dataset.
	"""
	if not exists(filepath):
		str_err = "Path " + filepath + " does not exist."
		print(str_err)
		sys.exit()

	index = kwargs.get("index_col", False)
	sep = kwargs.get("sep", ',')
	dtypes = kwargs.get("dtypes", None)
	print("Reading data file ...")
	if not isinstance(index, str):
		index = False
	if not isinstance(sep, str):
		sep = ','
	if not isinstance(dtypes, dict) and any([not isinstance(dtypes[key], str) for key in dtypes.keys()]):
		dct_types = None
	df = read_csv(filepath, index_col=index, sep=sep, dtype=dtypes)
	print("Data have been retrieved.")
	return df