import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from my_logistic_regression import MyLogisticMetrics as mylogrmetrics

# -- associated color to each house -- #
dct_palet = {"Hufflepuff":"dodgerblue",
			 "Gryffindor":"red",
			 "Slytherin":"green",
			 "Ravenclaw":"goldenrod"}

dct_metrics = {"accuracy" : mylogrmetrics.accuracy_score_,
			   "precision" : mylogrmetrics.precision_score_,
			   "recall" : mylogrmetrics.recall_score_,
			   "specificity" : mylogrmetrics.specificity_score_,
			   "F1" : mylogrmetrics.f1_score_}

# =========================================================================== #
#                        | Definition des fonctions |                         #
# =========================================================================== #
def static_plot(clfs, x_train, y_train, x_dev, y_dev):
	""" Graphical part of the project. The function displays:
		* The boundary decision of the One-vs-All model with the dataset in
		  the 2D_plan (Herbology - Def Against Dark Arts),
		* The cost functions of all the classifiers wrt iteration,
		* The accuracy and the recall of each classifiers.
	"""
	# -- Declaring the figure and the axes -- #
	fig = plt.figure(figsize=(15,9.5))
	gs = GridSpec(2, 2, figure=fig)
	axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
	
	# --formatting the different axes -- #
	axes[0].set_xlabel("Defense Against the Dark Arts")
	axes[0].set_ylabel("Herbology")
	axes[0].set_title("Decision boundary")
	axes[1].set_xlabel("i: iteration")
	axes[1].set_ylabel(r"$\mathcal{L}_{\theta_0,\theta_1}$")
	axes[1].grid()
	axes[2].set_xlabel("i: iteration")
	axes[2].set_ylabel("Scores (Accuracy & Recall)")
	axes[2].grid()
	axes[2].set_ylim(0.0,1.00)
	
	# --Plotting the boundary decision, cost function and Precision & Recall -- #
	plot_decision_boundary(clfs, x_train[["Defense Against the Dark Arts", "Herbology"]], y_train, axes[0])
	plot_cost_function_one_vs_all(clfs, x_train, y_train, axes[1])
	plot_scores(clfs, x_train, y_train, axes[2])

	# -- Displaying the plot -- #
	plt.show()


def plot_decision_boundary(model,X, Y, axe):
	# -- Set min and max values and give it some padding -- #
	X_min = []
	X_max = []
	for i in range(X.shape[1]):
		X_min.append(X.iloc[:,i].min() - 1)
		X_max.append(X.iloc[:,i].max() + 1)
	
	h = 0.01
	# -- Generate a grid of points with distance h between them -- #
	XX_1, XX_2 = np.meshgrid(np.arange(X_min[0], X_max[0], h),
							  np.arange(X_min[1], X_max[1], h))
	zeros_arr = np.zeros((XX_1.shape[0] * XX_1.shape[1], 1))
	# -- Predict the function value for the whole grid -- #
	XX = np.c_[XX_1.ravel(), XX_2.ravel(), zeros_arr.ravel(), zeros_arr.ravel(), zeros_arr.ravel()]
	class_preds = one_vs_all_prediction(model, XX)
	Z = _one_vs_all_class_onehot_(class_preds)
	Z = Z.reshape(XX_1.shape)
	
	# -- Plot the contour and training examples -- #
	axe.contourf(XX_1, XX_2, Z, 3, colors=["red", "green", "goldenrod", "dodgerblue"], alpha=0.5)
	lst_colors = np.array([dct_palet[house] for house in Y.values[:,0]])
	axe.scatter(X.values[:, 0], X.values[:, 1], c=lst_colors, edgecolor="k")


def one_vs_all_prediction(classifiers:list, x:np.array) -> np.array:
	raw_preds = _one_vs_all_predict_(classifiers, x)
	refine_preds = _one_vs_all_refine_predict_(classifiers, raw_preds)
	class_preds = _one_vs_all_class_attribution_(classifiers, refine_preds)
	return np.array(class_preds)


def _one_vs_all_predict_(classifiers:list, x:np.array) -> np.array:    
	# Predict using forward propagation and a classification threshold of 0.5
	res = None
	for clf in classifiers:
		vec = clf.predict_(x)
		if res is None:
			res = vec
		else:
			res = np.append(res, vec, 1)
	return res


def _one_vs_all_refine_predict_(classifiers:list, raw_pred:np.array) -> np.array:
	idx_max = np.argmax(raw_pred, axis = 1)
	refine_pred = np.zeros((raw_pred.shape[0], len(classifiers)))
	refine_pred[np.arange(0, refine_pred.shape[0]), idx_max] = 1
	return refine_pred


def _one_vs_all_class_attribution_(classifiers:list, refine_pred:np.array) -> np.ndarray:
	idx = np.argmax(refine_pred, axis = 1)
	house = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
	class_pred = [house[i] for i in idx]
	return class_pred


def _one_vs_all_class_onehot_(class_pred:np.array) -> np.ndarray:
	house = {"Gryffindor":1, "Slytherin":2, "Ravenclaw":3, "Hufflepuff":4}
	onehot_pred = np.array([house[pred] for pred in class_pred])
	return onehot_pred


def plot_cost_function_one_vs_all(models, x, y, axe):
	for model, i in zip(models, [1, 2, 3, 4]):
		idx = np.linspace(start=0, stop=model.max_iter, num=model.steps + 2)
		cost = model.cost_history_(x, y[f"target clf{i}"].values.reshape(-1,1))
		axe.plot(idx, cost.T, ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[model.tag])


def plot_scores(models, x, y, axe):
	for model, ii in zip(models, [1, 2, 3, 4]):
		y_ = y[f"target clf{ii}"].values.reshape(-1,1)
		idx = np.linspace(start=0, stop=model.max_iter, num=model.steps + 2)
		pred = np.round(model.predict_history_(x)).astype(int)
		acc_history = [mylogrmetrics.accuracy_score_(y_, pred[:,jj].reshape(-1,1)) for jj in range(pred.shape[1])]
		recall_history = [mylogrmetrics.recall_score_(y_, pred[:,jj].reshape(-1,1)) for jj in range(pred.shape[1])]
		axe.plot(idx, acc_history, ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[model.tag])
		axe.plot(idx, recall_history, ls='--', marker='o', ms=2, lw=1.2, color=dct_palet[model.tag])
