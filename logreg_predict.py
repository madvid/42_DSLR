import pandas as pd
import numpy as np
import sys

import tinystatistician as tinystat
import my_logistic_regression as mylogr
from utils import *

def checking_features(columns:list, lst_features:list) -> bool:
	"""
	Parameters:
		* columns [list]: list of columns in the dataframe.
		* lst_features [list]: list of required features.
	Return:
		True/False [bool]: If all / not all the required features
						   are in the dataframe. 
	"""
	if all([col in lst_features for col in columns]):
		return True
	else:
		return False

# --------------------------------------------------------------------------- #
# __________________________________ MAIN ___________________________________ #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
	# ==== ==
	# Parsing of arguments:
	# ==== ==
	b_visu = b_dynamic = b_static = b_console = False
	
	argv = sys.argv[1:]
	if len(argv) == 1 and (argv[0] in ["-h", "--help", "--usage"]):
		print_predict_usage()
		sys.exit()
	for arg in argv:
		if (arg == "--graphic=console") and (b_visu == False):
			b_visu = True
			b_console = True
		elif (arg == "--graphic=static") and (b_visu == False):
			b_visu = True
			b_static = True
		elif (arg == "--graphic=dynamic") and (b_visu == False):
			b_visu = True
			b_dynamic = True
		else:
			ag_dataset = arg.split('=')
			if len(ag_dataset) == 2:
				if ag_dataset[0] == "--dataset":
					...
				else:
					str_expt = "Dataset argument is incorrect."
					raise Exception(str_expt)
			elif arg not in lst_possible_args:
				str_expt = "Invalid argument."
				raise Exception(str_expt)
			else:
				str_expt = "Method or graphic argument cannot be define more than once."
				raise Exception(str_expt)
	
	if b_visu == False:
		b_console = True
	
	# ==== ==
	# Retrieve of the data from the csv file dedicated for the training:
	# ==== ==
	dct_types = {"Arithmancy" : np.float32,
				 "Astronomy" : np.float32,
				 "Herbology" : np.float32,
				 "Defense Against the Dark Arts" : np.float32,
				 "Divination" : np.float32,
				 "Muggle Studies" : np.float32,
				 "Ancient Runes" : np.float32,
				 "History of Magic" : np.float32,
				 "Transfiguration" : np.float32,
				 "Potions" : np.float32,
				 "Care of Magical Creatures" : np.float32,
				 "Charms" : np.float32,
				 "Flying" : np.float32}
	
	df = open_read_data("datasets/dataset_test.csv", index_col="Index", dtypes=dct_types)
	df.fillna(df.mean(),inplace=True)
	target = "Hogwarts House"
	lst_features = ["Herbology", "Divination", "Defense Against the Dark Arts",
					"History of Magic", "Ancient Runes"]
	nb_features = len(lst_features)

	if not checking_features(df.columns.values, lst_features):
		str_exit = "Issue with dataset of tests: possible missing features."
		print(str_exit)
		sys.exit()
	
	parameters = open_read_coeff(nb_features,4)
	if parameters is None:
		str_exit = "Issue with file: models.json. Quitting the program."
		print(str_exit)
		sys.exit()
	
	df_features = df[lst_features].copy()
	# ==== ==
	# Standardization of the numerical data in the differents
	# ==== ==
	tinystat.standardization(df_features)

	# ==== ==
	# Declaration and initialization of the logistic classifiers
	# ==== ==
	clf1 = mylogr.MyLogisticRegression(parameters["weights_1"],
									   b=parameters["biais_1"])
	clf2 = mylogr.MyLogisticRegression(parameters["weights_2"],
									   b=parameters["biais_2"])
	clf3 = mylogr.MyLogisticRegression(parameters["weights_3"],
									   b=parameters["biais_3"])
	clf4 = mylogr.MyLogisticRegression(parameters["weights_4"],
									   b=parameters["biais_4"])
	
	# ==== ==
	# Calcul of the probability via each classifiers
	# ==== ==
	df_res = pd.DataFrame()
	df_res["pred clf1"] = clf1.predict_(df_features.values).reshape(-1,)
	df_res["pred clf2"] = clf2.predict_(df_features.values).reshape(-1,)
	df_res["pred clf3"] = clf3.predict_(df_features.values).reshape(-1,)
	df_res["pred clf4"] = clf4.predict_(df_features.values).reshape(-1,)

	df_res.index.rename("Index", inplace=True)

	# ==== ==
	# Labels prediction based on probabilities of each classfiers
	# ==== ==
	pred_name = ["pred clf1", "pred clf2", "pred clf3", "pred clf4"]
	dct_pred = {"pred clf1":"Gryffindor",
				"pred clf2":"Slytherin",
				"pred clf3":"Ravenclaw",
				"pred clf4":"Hufflepuff"}
	
	df_res["Hogwarts House"] = df_res[pred_name].idxmax(axis="columns")
	df_res["Hogwarts House"].replace(to_replace=dct_pred, inplace=True)

	# ==== ==
	# Writing results into houses.csv
	# ==== ==
	df_res.to_csv("houses.csv", columns=["Hogwarts House"])