import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
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

# Variables for animated graphic
fps = 15
#interval = 8

# =========================================================================== #
#                        | Definition des fonctions |                         #
# =========================================================================== #
def static_plot(clfs, x_train, y_train, x_dev, y_dev):
	"""
	... Docstring ...
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
	axes[2].set_ylabel("Scores (Precision & Recall)")
	axes[2].grid()
	axes[2].set_ylim(0.0,1.00)
	
	# --Plotting the boundary decision, cost function and Precision & Recall -- #
	plot_decision_boundary(clfs, x_train[["Defense Against the Dark Arts", "Herbology"]], y_train, axes[0])
	#plt.yscale("log")
	plot_cost_function_one_vs_all(clfs, x_train, y_train, axes[1])
	plot_scores(clfs, x_train, y_train, axes[2])

	# -- Displaying the plot -- #
	plt.show()


def plot_decision_boundary(model,X, Y, axe):
	"""
	... Docstring ...
	"""
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
	"""
	... Docstring ...
	"""
	raw_preds = _one_vs_all_predict_(classifiers, x)
	refine_preds = _one_vs_all_refine_predict_(classifiers, raw_preds)
	class_preds = _one_vs_all_class_attribution_(classifiers, refine_preds)
	return np.array(class_preds)


def _one_vs_all_predict_(classifiers:list, x:np.array) -> np.array:    
	"""
	... Docstring ...
	"""
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
	"""
	... Docstring ...
	"""
	idx_max = np.argmax(raw_pred, axis = 1)
	refine_pred = np.zeros((raw_pred.shape[0], len(classifiers)))
	refine_pred[np.arange(0, refine_pred.shape[0]), idx_max] = 1
	return refine_pred


def _one_vs_all_class_attribution_(classifiers:list, refine_pred:np.array) -> np.ndarray:
	"""
	... Docstring ...
	"""
	idx = np.argmax(refine_pred, axis = 1)
	house = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
	class_pred = [house[i] for i in idx]
	return class_pred


def _one_vs_all_class_onehot_(class_pred:np.array) -> np.ndarray:
	"""
	... Docstring ...
	"""
	house = {"Gryffindor":1, "Slytherin":2, "Ravenclaw":3, "Hufflepuff":4}
	onehot_pred = np.array([house[pred] for pred in class_pred])
	return onehot_pred


def plot_cost_function_one_vs_all(models, x, y, axe):
	"""
	... Docstring ...
	"""

	for model, i in zip(models, [1, 2, 3, 4]):
		idx = np.linspace(start=0, stop=model.max_iter, num=model.steps + 2)
		cost = model.cost_history_(x, y[f"target clf{i}"].values.reshape(-1,1))
		axe.plot(idx, cost.T, ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[model.tag])


def plot_scores(models, x, y, axe):
	"""
	... Docstring ...
	"""
	for model, ii in zip(models, [1, 2, 3, 4]):
		y_ = y[f"target clf{ii}"].values.reshape(-1,1)
		idx = np.linspace(start=0, stop=model.max_iter, num=model.steps + 2)
		pred = np.round(model.predict_history_(x)).astype(int)
		acc_history = [mylogrmetrics.accuracy_score_(y_, pred[:,jj].reshape(-1,1)) for jj in range(pred.shape[1])]
		recall_history = [mylogrmetrics.recall_score_(y_, pred[:,jj].reshape(-1,1)) for jj in range(pred.shape[1])]
		axe.plot(idx, acc_history, ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[model.tag])
		axe.plot(idx, recall_history, ls='--', marker='o', ms=2, lw=1.2, color=dct_palet[model.tag])

class DynamicVisu():
	def __init__(self, clfs, x, y):
		"""
		"""
		self.clfs = clfs
		self.x = x
		self.y = y
		
		plt.ion()
		self.fig = plt.figure(figsize=(15,9.5))
		self.fig.canvas.mpl_connect('close_event', self.f_close)
		gs = GridSpec(2, 2, figure=self.fig)
		self.axes = [self.fig.add_subplot(gs[:, 0]), self.fig.add_subplot(gs[0, 1]), self.fig.add_subplot(gs[1, 1])]
		self.formating()
		
		self.idx = np.array([0])
		self.cost_clf1 = self.clfs[0].cost_(self.x, self.y[f"target clf1"].values.reshape(-1,1))
		self.cost_clf2 = self.clfs[1].cost_(self.x, self.y[f"target clf2"].values.reshape(-1,1))
		self.cost_clf3 = self.clfs[2].cost_(self.x, self.y[f"target clf3"].values.reshape(-1,1))
		self.cost_clf4 = self.clfs[3].cost_(self.x, self.y[f"target clf4"].values.reshape(-1,1))
		self.prec_clf1 = np.array([0])
		self.prec_clf2 = np.array([0])
		self.prec_clf3 = np.array([0])
		self.prec_clf4 = np.array([0])
		self.recall_clf1 = np.array([0])
		self.recall_clf2 = np.array([0])
		self.recall_clf3 = np.array([0])
		self.recall_clf4 = np.array([0])
		self._grid_for_boundary_()
		
		self.pred_on_mesgrid()


	def formating(self):
		"""
		"""
		self.axes[0].set_xlabel("Defense Against the Dark Arts")
		self.axes[0].set_ylabel("Herbology")
		self.axes[0].set_title("Decision boundary")
		
		self.axes[1].set_xlabel("i: iteration")
		self.axes[1].set_xlim(-10, 10000)
		self.axes[1].set_ylim(0, 5)
		self.axes[1].set_ylabel(r"$\mathcal{L}_{\theta_0,\theta_1}$")
		self.axes[1].grid()
		
		self.axes[2].set_xlabel("i: iteration")
		self.axes[2].set_ylabel("Scores (Precision & Recall)")
		self.axes[2].set_xlim(-10, 10000)
		self.axes[2].set_ylim(0, 1.01)
		self.axes[2].grid()
		self.axes[2].set_ylim(0.0,1.00)

	
	def _grid_for_boundary_(self):
		"""
		"""
		X = self.x[["Defense Against the Dark Arts", "Herbology"]]
		Y = self.y.values.reshape(-1,1)
		X_min, X_max = [], []
		for i in range(X.shape[1]):
			X_min.append(X.iloc[:,i].min() - 1)
			X_max.append(X.iloc[:,i].max() + 1)
	
		h = 0.01
		# -- Generate a grid of points with distance h between them -- #
		self.XX_1, self.XX_2 = np.meshgrid(np.arange(X_min[0], X_max[0], h),
							     np.arange(X_min[1], X_max[1], h))
		zeros_arr = np.zeros((self.XX_1.shape[0] * self.XX_1.shape[1], 1))
		self.XX = np.c_[self.XX_1.ravel(), self.XX_2.ravel(),
				   zeros_arr.ravel(), zeros_arr.ravel(), zeros_arr.ravel()]


	def pred_on_mesgrid(self):
		"""
		"""
		raw_preds = _one_vs_all_predict_(self.clfs, self.XX)
		refine_preds = _one_vs_all_refine_predict_(self.clfs, raw_preds)
		class_preds = _one_vs_all_class_attribution_(self.clfs, refine_preds)
		Z = _one_vs_all_class_onehot_(class_preds)
		self.Z = Z.reshape(self.XX_1.shape)

	
	def init_anim(self):
		"""
		"""
		self.boundary = self.axes[0].contourf(self.XX_1, self.XX_2, self.Z, 3,
								colors=["red", "green", "goldenrod", "dodgerblue"], alpha=0.5)
	
		lst_colors = np.array([dct_palet[house] for house in self.y.values[:,0]])
		self.raw_data = self.axes[0].scatter(self.x["Defense Against the Dark Arts"].values,
								   self.x["Herbology"].values, c=lst_colors, edgecolor="k")

		## Initialisation of the Line2D object for the Axes[1] objects
		self.l_cost_clf1, = self.axes[1].plot(self.idx, self.cost_clf1,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[0].tag])
		self.l_cost_clf2, = self.axes[1].plot(self.idx, self.cost_clf2,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[1].tag])
		self.l_cost_clf3, = self.axes[1].plot(self.idx, self.cost_clf3,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[2].tag])
		self.l_cost_clf4, = self.axes[1].plot(self.idx, self.cost_clf4,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[3].tag])

		## Initialisation of the Line2D object for the Axes[2] objects
		self.l_prec_clf1, = self.axes[2].plot(self.idx, self.prec_clf1,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[0].tag])
		self.l_prec_clf2, = self.axes[2].plot(self.idx, self.prec_clf2,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[1].tag])
		self.l_prec_clf3, = self.axes[2].plot(self.idx, self.prec_clf3,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[2].tag])
		self.l_prec_clf4, = self.axes[2].plot(self.idx, self.prec_clf4,
									ls='-', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[3].tag])
		self.l_recall_clf1, = self.axes[2].plot(self.idx, self.recall_clf1,
									ls='--', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[0].tag])
		self.l_recall_clf2, = self.axes[2].plot(self.idx, self.recall_clf2,
									ls='--', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[1].tag])
		self.l_recall_clf3, = self.axes[2].plot(self.idx, self.recall_clf3,
									ls='--', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[2].tag])
		self.l_recall_clf4, = self.axes[2].plot(self.idx, self.recall_clf4,
									ls='--', marker='o', ms=2, lw=1.2, color=dct_palet[self.clfs[3].tag])

		
	def update(self, i):
		"""
		"""
		n_cycle = 100
		self.clfs[0].fit_(self.x.values, self.y["target clf1"].values.reshape(-1,1), n_cycle)
		self.clfs[1].fit_(self.x.values, self.y["target clf2"].values.reshape(-1,1), n_cycle)
		self.clfs[2].fit_(self.x.values, self.y["target clf3"].values.reshape(-1,1), n_cycle)
		self.clfs[3].fit_(self.x.values, self.y["target clf4"].values.reshape(-1,1), n_cycle)

		self.pred_on_mesgrid()
		self.idx = np.concatenate((self.idx, np.array([i * n_cycle])))
	
		self.cost_clf1 = np.concatenate((self.cost_clf1, self.clfs[0].cost_(self.x.values, self.y["target clf1"].values.reshape(-1,1))))
		self.cost_clf2 = np.concatenate((self.cost_clf2, self.clfs[1].cost_(self.x.values, self.y["target clf2"].values.reshape(-1,1))))
		self.cost_clf3 = np.concatenate((self.cost_clf3, self.clfs[2].cost_(self.x.values, self.y["target clf3"].values.reshape(-1,1))))
		self.cost_clf4 = np.concatenate((self.cost_clf4, self.clfs[3].cost_(self.x.values, self.y["target clf4"].values.reshape(-1,1))))

		self.y["predicted Hogwarts House"] = one_vs_all_prediction(self.clfs, self.x)
		yhat =  self.y["predicted Hogwarts House"].values.reshape(-1,1)
		y = self.y["Hogwarts House"].values.reshape(-1,1)
		tmp_prec_1 = dct_metrics["precision"](y, yhat, "Gryffindor")
		tmp_prec_2 = dct_metrics["precision"](y, yhat, "Slytherin")
		tmp_prec_3 = dct_metrics["precision"](y, yhat, "Ravenclaw")
		tmp_prec_4 = dct_metrics["precision"](y, yhat, "Hufflepuff")
		tmp_recall_1 = dct_metrics["recall"](y, yhat, "Gryffindor")
		tmp_recall_2 = dct_metrics["recall"](y, yhat, "Slytherin")
		tmp_recall_3 = dct_metrics["recall"](y, yhat, "Ravenclaw")
		tmp_recall_4 = dct_metrics["recall"](y, yhat, "Hufflepuff")

		self.prec_clf1 = np.concatenate((self.prec_clf1, np.array([tmp_prec_1])))
		self.prec_clf2 = np.concatenate((self.prec_clf2, np.array([tmp_prec_2])))
		self.prec_clf3 = np.concatenate((self.prec_clf3, np.array([tmp_prec_3])))
		self.prec_clf4 = np.concatenate((self.prec_clf4, np.array([tmp_prec_4])))
		self.recall_clf1 = np.concatenate((self.recall_clf1, np.array([tmp_recall_1])))
		self.recall_clf2 = np.concatenate((self.recall_clf2, np.array([tmp_recall_2])))
		self.recall_clf3 = np.concatenate((self.recall_clf3, np.array([tmp_recall_3])))
		self.recall_clf4 = np.concatenate((self.recall_clf4, np.array([tmp_recall_4])))

		for coll in self.boundary.collections:
		# Remove the existing contours
			coll.remove()
			#self.boundary.collections.remove(coll)

		self.boundary = self.axes[0].contourf(self.XX_1, self.XX_2, self.Z, 3, colors=["red", "green", "goldenrod", "dodgerblue"], alpha=0.5)
		#self.raw_data = axes[0].scatter(x_train["Defense Against the Dark Arts"].values, x_train["Herbology"].values, c=lst_colors, edgecolor="k")
		self.l_cost_clf1.set_data(self.idx, self.cost_clf1)
		self.l_cost_clf2.set_data(self.idx, self.cost_clf2)
		self.l_cost_clf3.set_data(self.idx, self.cost_clf3)
		self.l_cost_clf4.set_data(self.idx, self.cost_clf4)

		self.l_prec_clf1.set_data(self.idx, self.prec_clf1)
		self.l_prec_clf2.set_data(self.idx, self.prec_clf2)
		self.l_prec_clf3.set_data(self.idx, self.prec_clf3)
		self.l_prec_clf4.set_data(self.idx, self.prec_clf4)
		self.l_recall_clf1.set_data(self.idx, self.recall_clf1)
		self.l_recall_clf2.set_data(self.idx, self.recall_clf2)
		self.l_recall_clf3.set_data(self.idx, self.recall_clf3)
		self.l_recall_clf4.set_data(self.idx, self.recall_clf4)

	def anim(self):
		"""
		"""
		self.anim_fig = FuncAnimation(self.fig, self.update, repeat=False)
		plt.show()
		

	def f_close(self, event):
		""" Functions called when the graphical window is closed.
		It prints the last value of the theta vector and the last value of the
		cost function.
		"""
		
