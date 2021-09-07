# =========================================================================== #
#                       |Importation des lib/packages|                        #
# =========================================================================== #
# --- librairies standards --- #
import pandas as pd
import numpy as np
import sys

# --- librairies locales --- #
import tinystatistician as tinystat
from my_logistic_regression import MyLogisticRegression as mylogr
from my_logistic_regression import MyLogisticRegressionWithHistory as mylogrwh
from my_logistic_regression import MyLogisticMetrics as mylogrmetrics
from parsing import parser
from utils import *
from utils_graphic import *

# =========================================================================== #
#                        | Definition des constantes|                         #
# =========================================================================== #

# Les series d'intérêt dans le dataset de train ou de test sont les clefs du dct
# suivant. Nous fixons le type de ces series
dct_types = {"Arithmancy" : np.float32, "Astronomy" : np.float32,
				 "Herbology" : np.float32, "Defense Against the Dark Arts" : np.float32,
				 "Divination" : np.float32, "Muggle Studies" : np.float32,
				 "Ancient Runes" : np.float32, "History of Magic" : np.float32,
				 "Transfiguration" : np.float32, "Potions" : np.float32,
				 "Care of Magical Creatures" : np.float32, "Charms" : np.float32,
				 "Flying" : np.float32}

# -- values to predict -- #
target = "Hogwarts House"

# -- features used in the models -- #
lst_features = ["Arithmancy", "Herbology", "Defense Against the Dark Arts", "Divination",
					"Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration"]

# -- training / dev set ratio -- #
split_ratio = 0.7

# =========================================================================== #
#                        | Definition des fonctions |                         #
# =========================================================================== #
def score_report(df:pd.DataFrame, metrics:list):
	""" Prints a report on the score with the different metrics for all
	of the classifiers (predictions in the dataframe)
	
	Parameters:
		* df[pandas.DataFrame]: datafram containing the true label and the
								predicted labels of the different classifiers
		* metrics [list]: list of the different metrics to display in the
						  report for the different classifiers
	"""
	acceptable_metrics = ["accuracy",
						  "precision",
						  "recall",
						  "specificity",
						  "F1",
						  "Confusion matrix"] 
	if not all([metric in acceptable_metrics for metric in metrics]):
		str_expt = "At least one element in the list metrics is incorrect "\
				 + "or not handled."
		raise Exception(str_expt)

	index = ["Gryffindor",
			 "Slytherin",
			 "Ravenclaw",
			 "Hufflepuff"]
	columns = metrics + ["size"]
	df_report = pd.DataFrame(data=np.zeros((len(index), len(columns))),
							 index=index, columns=columns)

	for house in index:
		for metric in metrics:
			if metric in ["accuracy", "precision", "recall", "F1"]:
				val = dct_metrics[metric](df["Hogwarts House"].values.reshape(-1,1),
										  df["predicted Hogwarts House"].values.reshape(-1,1), house)
			df_report.loc[house, metric] = val
		df_report.loc[house, "size"] = df[df["predicted Hogwarts House"] == house].shape[0]
	
	print("\nScore metrics report:\n", df_report)
	if "Confusion matrix" in metrics:
		print("\nConfusion matrix:\n",
			  mylogrmetrics.confusion_matrix_(df_res["Hogwarts House"].values.reshape(-1,1),
											  df_res["predicted Hogwarts House"].values.reshape(-1,1)))


# --------------------------------------------------------------------------- #
# Functions definition related to the animated visualization                  #
# --------------------------------------------------------------------------- #
def animate_visu(clfs, x_train, y_train):
	plt.style.use('seaborn-pastel')

	dyn_visu = DynamicVisu(clfs, x_train, y_train)
	dyn_visu.init_anim()
	dyn_visu.anim()


# =========================================================================== #
# _________________________________  MAIN  __________________________________ #
# =========================================================================== #

if __name__ == "__main__":
	# --- Parsing of the arguments -- #
	argv = sys.argv[1:]
	datapath, b_visu, b_static, b_console, b_gd, b_sgd, b_minibatch, b_method = parser(argv)
	
	# -- Reading of the data from the csv file dedicated for the training -- #
	df = open_read_data(datapath, index_col="Index", dtypes=dct_types)

	df.fillna(df.mean(),inplace=True)
	nb_features = len(lst_features)
	df_m1 = df[lst_features + [target]].copy()
	
	# -- Standardization of the numerical data in the differents -- #
	tinystat.standardization(df_m1)
	
	# -- Shuffling and splitting data into train and dev sets -- #
	df_m1 = df_m1.sample(frac = 1)
	df_train = df_m1[:int(split_ratio * df_m1.shape[0])]
	df_dev = df_m1[int(split_ratio * df_m1.shape[0]):]
	
	# ==== ==
	# Separation of x and y data of the training set
	# and preparation of the target of each classifiers
	# ==== ==
	x_train = df_train[lst_features].copy()
	y_train = df_train[[target]].copy()

	y_train[["target clf1", "target clf2", "target clf3", "target clf4"]]= 0
	y_train.loc[y_train[target] == "Gryffindor", "target clf1"] = 1
	y_train.loc[y_train[target] == "Slytherin", "target clf2"] = 1
	y_train.loc[y_train[target] == "Ravenclaw", "target clf3"] = 1
	y_train.loc[y_train[target] == "Hufflepuff", "target clf4"] = 1

	# ==== ==
	# Separation of x and y data of the dev set
	# and preparation of the target of each classifiers
	# ==== ==
	x_dev = df_dev[lst_features].copy()
	y_dev = df_dev[[target]].copy()

	y_dev[["target clf1", "target clf2", "target clf3", "target clf4"]]= 0
	y_dev.loc[y_dev[target] == "Gryffindor", "target clf1"] = 1
	y_dev.loc[y_dev[target] == "Slytherin", "target clf2"] = 1
	y_dev.loc[y_dev[target] == "Ravenclaw", "target clf3"] = 1
	y_dev.loc[y_dev[target] == "Hufflepuff", "target clf4"] = 1
	
	# ==== ==
	# Declaration and initialization of the logistic classifiers
	# ==== ==
	if b_console:
		model = mylogr
	if b_static:
		model = mylogrwh
	clf1 = model(np.random.rand(nb_features, 1),
				  alpha=1e-2,
				  max_iter=10000,
				  lambd=0.1, tag="Gryffindor", method="minibatch+momentum")
	clf2 = model(np.random.rand(nb_features, 1),
				  alpha=1e-2,
				  max_iter=10000,
				  lambd=0.1, tag="Slytherin", method="minibatch+momentum")
	clf3 = model(np.random.rand(nb_features, 1),
				  alpha=1e-2,
				  max_iter=10000,
				  lambd=0.1, tag="Ravenclaw", method="minibatch+momentum")
	clf4 = model(np.random.rand(nb_features, 1),
				  alpha=1e-2,
				  max_iter=10000,
				  lambd=0.1, tag="Hufflepuff", method="minibatch+momentum")
	clfs = [clf1, clf2, clf3, clf4]
	
	# ==== ==
	# Fitting process for all logistic classifiers
	# ==== ==
	if b_gd:
		# -- classic gradient descent -- #
		i = 1
		for clf in clfs:
			if b_static:
				clf.fit_history_(x_train.values, y_train[f"target clf{i}"].values.reshape(-1,1))
			else:
				clf.fit_(x_train.values, y_train[f"target clf{i}"].values.reshape(-1,1))
			i += 1
	elif b_sgd:
		# -- stochastic gradient descent -- #
		i = 1
		for clf in clfs:
			if b_static:
				clf.stochastic_fit_history_(x_train.values, y_train[f"target clf{i}"].values.reshape(-1,1))
			else:
				clf.stochastic_fit_(x_train.values, y_train[f"target clf{i}"].values.reshape(-1,1))
			i += 1
	elif b_minibatch:
		# -- minibatch gradient descent -- #
		mini_batches_clfs = []
		for i in range(4):
			mini_batches_clfs.append(mylogr.init_mini_batches(x_train.values,
									 y_train[f"target clf{i+1}"].values.reshape(-1,1)))

		if b_static:
			for clf, mini_batches_clf in zip(clfs, mini_batches_clfs):
				clf.minibatch_fit_history_(mini_batches_clf)
		else:
			for clf, mini_batches_clf in zip(clfs, mini_batches_clfs):
				clf.minibatch_fit_(mini_batches_clf)

	# -- Saving the weights and biais of each classifier -- #
	weights = np.concatenate((clf1.theta, clf2.theta, clf3.theta, clf4.theta), axis=1).T
	b = np.array([clf1.b, clf2.b, clf3.b, clf4.b])
	open_write_coeff(weights, b)

	# -- Prediction of the target on the entire test set -- #
	df_res = y_dev.copy()
	df_res["pred clf1"] = clf1.predict_(x_dev.values)
	df_res["pred clf2"] = clf2.predict_(x_dev.values)
	df_res["pred clf3"] = clf3.predict_(x_dev.values)
	df_res["pred clf4"] = clf4.predict_(x_dev.values)
	
	df_res = df_res.reindex(columns=["Hogwarts House",
									 "target clf1", "pred clf1",
					 				 "target clf2", "pred clf2",
									 "target clf3", "pred clf3",
									 "target clf4", "pred clf4"])
	
	pred_name = ["pred clf1", "pred clf2", "pred clf3", "pred clf4"]
	dct_pred = {"pred clf1":"Gryffindor",
				"pred clf2":"Slytherin",
				"pred clf3":"Ravenclaw",
				"pred clf4":"Hufflepuff"}
	
	s_pred_label = df_res[pred_name].idxmax(axis="columns").replace(to_replace=dct_pred)
	df_res["predicted Hogwarts House"] = s_pred_label
	
	# ==== ==
	# Printing some metrics on the dev set.
	# * In case of console mode, only score is printed.
	# * In case of graphic=static, score is printed + a plot to see performance
	# 	along the training phase.
	# * In case of graphic=dynamic, score is printed + a dynamic plot to see
	#	performance in "real time".
	# ==== ==
	score_report(df_res, ["accuracy", "precision", "recall", "F1", "Confusion matrix"])

	if b_static:
		#print(y_train.head(20))
		static_plot(clfs, x_train, y_train, x_dev, y_dev)