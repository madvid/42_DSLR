# =========================================================================== #
#                       |Importation des lib/packages|                        #
# =========================================================================== #
# --- librairies standards --- #
import pandas as pd
import numpy as np
import sys

# --- librairies locales --- #
import tinystatistician as tinystat
import my_logistic_regression as mylogr
from utils import *
from parsing import parser_predict

# =========================================================================== #
#                        | Definition des constantes|                         #
# =========================================================================== #
from constants import dct_types, target, lst_features, nb_features, RED, END

# =========================================================================== #
#                        | Definition des fonctions |                         #
# =========================================================================== #
def checking_features(columns:list, lst_features:list) -> bool:
	"""
	Parameters:
		* columns [list]: list of columns in the dataframe.
		* lst_features [list]: list of required features.
	Return:
		True/False [bool]: If all / not all the required features
						   are in the dataframe. 
	"""
	if all([feat in columns for feat in lst_features]):
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
	argv = sys.argv[1:]
	datapath, modelpath = parser_predict(argv)

	# ==== ==
	# Retrieve of the data from the csv file dedicated for the prediction:
	# ==== ==
	# -- Reading of the data from the csv file and checking
	df = open_read_file(datapath, "predict")
	if df is None:
		str_exit = RED + f"Issue with file: {datapath}. Quitting the program." + END
		print(str_exit)
		sys.exit()
	df.fillna(df.mean(),inplace=True)
	
	if not checking_features(df.columns.values, lst_features):
		str_exit = "Issue with dataset of tests: possible missing features."
		print(str_exit)
		sys.exit()
	
	# -- Reading the parameters of the One-vs-all model
	parameters = open_read_coeff(modelpath, nb_features, 4)
	if parameters is None:
		str_exit = RED + f"Issue with file: {modelpath}. Quitting the program." + END
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