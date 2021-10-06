import numpy as np
import pandas as pd

# small value to avoid ZeroDivisionError or RuntimeWarning due to invalid value encountered in long_scalars
eps = 1e-8
class MyLogisticRegression():
	"""
	...Docstring...
	"""
	def __init__(self, theta:np.ndarray, b:float=0.0,
		alpha:float=1e-4, max_iter:int=1000, lambd:float=0.0,
		tag:str="", gamma:float=0.9, method:str="GD"):
		if (theta.ndim != 2) or theta.shape[1] != 1:
			str_err = "theta must be of shape (n,1) where n is the number of features."
			raise Exception(str_err)
		if alpha <= 0 or alpha >= 1:
			str_err = "alpha must be a positive float less than 1."
			raise Exception(str_err)
		if max_iter <= 0:
			str_err = "max_iter must be a positive integer."
			raise Exception(str_err)
		if lambd < 0.0:
			str_err = "lambd must be a positive or null."
			raise Exception(str_err)
		self.tag = tag
		self.theta = theta
		self.b = b
		self.alpha = alpha
		self.max_iter = max_iter
		# -- For optimization with regularization -- #
		self.lambd = lambd
		# -- For SGD with momentum -- #
		self.gamma = gamma
		if ((self.gamma > 0) and (self.gamma < 1)):
			self.past_dtheta = np.zeros(theta.shape)
			self.past_db = 0.0
		else:
			self.gamma = 0.0
		
		if not method in ["GD", "SGD", "SGD+momentum", "minibatch"]:
			self.method = "GD"
		else:
			self.method = method
		
		if self.method == "GD":
			self.fit_ = self._fit_
		elif self.method == "SGD":
			self.fit_ = self._stochastic_fit_
		elif self.method == "minibatch":
			self.fit_ = self._minibatch_fit_
		elif self.method == "SGD+momentum":
			self.fit_ = self._stochastic_fit_w_momentum_
	
	
	def predict_(self, x:np.ndarray):
		"""
		...Docstring...
		"""
		if x.shape[1] != self.theta.shape[0]:
			str_err = "Mismatching shape between x and theta."
			raise Exception(str_err)
		z = (np.dot(x, self.theta) + self.b).astype(float)
		pred = np.divide (1.0, 1.0 + np.exp(-z))
		return (pred)


	def grad_(self, x:np.ndarray, y:np.ndarray):
		"""
		...Docstring...
		"""
		if x.ndim != 2:
			str_err = "Incorrect dimension for x."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if x.shape[1] != self.theta.shape[0]:
			str_err = "Mismatching shape between x and theta."
			raise Exception(str_err)
		if x.shape[0] != y.shape[0]:
			str_err = "Mismatching shape between x and y."
			raise Exception(str_err)
		if y.shape[1] != 1:
			str_err = "Incorrect y shape: y must be (m, 1) shaped."
			raise Exception(str_err)
		m = x.shape[0]
		pred = self.predict_(x)
		diff = pred - y
		dJ0 = (1 / m) * np.sum(diff, axis=0, keepdims=True) 
		dJ = (1 / m) * (np.sum(np.multiply(diff, x), axis=0, keepdims=True) + self.lambd * self.theta.T)
		dJ = np.concatenate((dJ0, dJ), axis=1)
		return dJ.reshape(-1,1)

	
	def _fit_(self, x:np.ndarray, y:np.ndarray, n_cycle=None):
		"""
		...Docstring...
		"""
		if type(n_cycle) is type(None):
			n_cycle = self.max_iter
		for _ in range(n_cycle):
			dJ = self.grad_(x,y)
			self.b = self.b - self.alpha * dJ[0]
			self.theta = self.theta - self.alpha * dJ[1:]
		return (self.theta, self.b)


	def _stochastic_fit_(self, x:np.ndarray, y:np.ndarray, n_cycle=None):
		"""
		...Docstring...
		"""
		m = x.shape[0]
		if type(n_cycle) is type(None):
			n_cycle = self.max_iter
		for ii in range(n_cycle):
			dJ = self.grad_(x[ii % m,:].reshape(1,-1),y[ii % m,:].reshape(1,-1))
			self.b = self.b - self.alpha * dJ[0]
			self.theta = self.theta - self.alpha * dJ[1:]
		return (self.theta, self.b)


	@staticmethod
	def init_mini_batches(X:np.ndarray, Y:np.ndarray, mini_batch_size=64, seed=0):
		"""
		...Docstring...
		"""
		np.random.seed(seed)
		m = X.shape[0]
		mini_batches = []
		permutation = np.random.permutation(m) # random sequence of int, without repetition
		X_shuffled = X[permutation]
		Y_shuffled = Y[permutation]
		complete_mbatch = np.math.floor(m / mini_batch_size) # nb of complete batch

		for ii in range(complete_mbatch):
			mini_batch_X = X_shuffled[ii * mini_batch_size : (ii + 1) * mini_batch_size, :]
			mini_batch_Y = Y_shuffled[ii * mini_batch_size : (ii + 1) * mini_batch_size, :]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		
		if (m % mini_batch_size) != 0:
			mini_batch_X = X_shuffled[(ii + 1) * mini_batch_size:, :]
			mini_batch_Y = Y_shuffled[(ii + 1) * mini_batch_size:, :]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		
		return mini_batches


	def _minibatch_fit_(self, mini_batches:list, n_cycle=None):
		"""
		...Docstring...
		"""
		n_mini_batches = len(mini_batches)
		self.n_mini_batches = n_mini_batches
		if type(n_cycle) is type(None):
			n_cycle = self.max_iter
		for _ in range(n_cycle):
			r = np.random.randint(n_mini_batches)
			X_minibatch, Y_minibatch = mini_batches[r]
			dJ = self.grad_(X_minibatch, Y_minibatch)
			self.b = self.b - self.alpha * dJ[0]
			self.theta = self.theta - self.alpha * dJ[1:]
		return (self.theta, self.b)


	def _stochastic_fit_w_momentum_(self, x:np.ndarray, y:np.ndarray, n_cycle=None):
		
		"""
		...Docstring...
		"""
		m = x.shape[0]
		if type(n_cycle) is type(None):
			n_cycle = self.max_iter
		for ii in range(n_cycle):
			dJ = self.grad_(x[ii % m,:].reshape(1,-1),y[ii % m,:].reshape(1,-1))
			self.b = self.b - self.alpha * dJ[0] - self.gamma * self.past_db
			self.theta = self.theta - self.alpha * dJ[1:] - self.gamma * self.past_dtheta
			self.past_db, self.past_dtheta = dJ[0], dJ[1:]
		return (self.theta, self.b)


	def cost_(self, x:np.ndarray, y:np.ndarray):
		"""
		... Docstring ...
		"""
		epsilon = 1e-5
		if x.ndim != 2:
			str_err = "Incorrect dimension for x."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if x.shape[1] != self.theta.shape[0]:
			str_err = "Mismatching shape between x and theta."
			raise Exception(str_err)
		if x.shape[0] != y.shape[0]:
			str_err = "Mismatching shape between x and y."
			raise Exception(str_err)
		if y.shape[1] != 1:
			str_err = "Incorrect y shape: y must be (m, 1) shaped."
			raise Exception(str_err)
		m = x.shape[0]
		pred = self.predict_(x)
		cost = -(1/m) * (np.multiply(y, np.log(pred + epsilon)) + np.multiply(1 - y, np.log(1 - pred + epsilon)))
		cost = np.sum(cost, axis=0) + (0.5 / m) * self.lambd * np.sum(np.square(self.theta))
		return cost


class MyLogisticRegressionWithHistory(MyLogisticRegression):
	"""
	...Docstring...
	"""
	def __init__(self, theta, b:float=0,
		alpha:float=1e-4, max_iter:int=1000, lambd:float=0.0,
		steps=100, tag:str ="", gamma:float=0.9, method:str="GD"):
		MyLogisticRegression.__init__(self, theta, b, alpha, max_iter, lambd, tag, gamma, method)
		self.steps = steps
		self.remind_iter = max_iter % int(max_iter / steps)
		self.cycle_per_step = int(max_iter / steps)
		self.theta_history = theta
		self.b_history = b

		if not method in ["GD", "SGD", "SGD+momentum", "minibatch"]:
			self.method = "GD"
		else:
			self.method = method
		
		if self.method == "GD":
			self.fit_history_ = self._fit_history_
		elif self.method == "SGD":
			self.fit_history_ = self._stochastic_fit_history_
		elif self.method == "SGD+momentum":
			self.fit_history_ = self._stochastic_fit_w_momentum_history_
		elif self.method == "minibatch":
			self.fit_history_ = self._minibatch_fit_history_


	def predict_history_(self, x:np.ndarray):
		"""
		... Docstring ...
		"""
		z = np.dot(x, self.theta_history) + self.b_history
		pred = np.divide(1.0, 1.0 + np.exp(-z))
		return pred


	def _stochastic_fit_(self, x:np.ndarray, y:np.ndarray, start:int, n_cycle=None):
		"""
		...Docstring...
		"""
		m = x.shape[0]
		if type(n_cycle) is type(None):
			n_cycle = self.max_iter
		for ii in range(n_cycle):
			dJ = self.grad_(x[(start + ii) % m,:].reshape(1,-1),y[(start + ii) % m,:].reshape(1,-1))
			self.b = self.b - self.alpha * dJ[0]
			self.theta = self.theta - self.alpha * dJ[1:]
		return (self.theta, self.b)


	def _stochastic_fit_w_momentum_(self, x:np.ndarray, y:np.ndarray, start:int, n_cycle=None):
		"""
		...Docstring...
		"""
		m = x.shape[0]
		if type(n_cycle) is type(None):
			n_cycle = self.max_iter
		for ii in range(n_cycle):
			dJ = self.grad_(x[(start + ii) % m,:].reshape(1,-1),y[(start + ii) % m,:].reshape(1,-1))
			self.b = self.b - self.alpha * dJ[0] - self.gamma * self.past_db
			self.theta = self.theta - self.alpha * dJ[1:] - self.gamma * self.past_dtheta
			self.past_db, self.past_dtheta = dJ[0], dJ[1:]
		return (self.theta, self.b)


	def _fit_history_(self, x:np.ndarray, y:np.ndarray):
		"""
		...Docstring...
		"""
		for _ in  range(self.steps):
			theta_b = self.fit_(x, y, self.cycle_per_step)
			self.theta_history = np.hstack((self.theta_history, theta_b[0]))
			self.b_history = np.hstack((self.b_history, theta_b[1]))
		
		# -- Performing the last iterations -- #
		theta_b = self.fit_(x,y, self.remind_iter)
		self.theta_history = np.hstack((self.theta_history, theta_b[0]))
		self.b_history = np.hstack((self.b_history, theta_b[1]))
		
		return (self.theta, self.b)


	def _stochastic_fit_history_(self, x:np.ndarray, y:np.ndarray):
		"""
		...Docstring...
		"""
		for ii in range(self.steps):
			theta_b = self.fit_(x, y, start=ii * self.cycle_per_step, n_cycle=self.cycle_per_step)
			self.theta_history = np.hstack((self.theta_history, theta_b[0]))
			self.b_history = np.hstack((self.b_history, theta_b[1]))

		# -- Performing the last iterations -- #
		theta_b = self.fit_(x, y, start=ii * self.cycle_per_step, n_cycle=self.remind_iter)
		self.theta_history = np.hstack((self.theta_history, theta_b[0]))
		self.b_history = np.hstack((self.b_history, theta_b[1]))
		
		return (self.theta, self.b)


	def _minibatch_fit_history_(self, mini_batches:list, n_cycle=None):
		"""
		...Docstring...
		"""
		for _ in  range(self.steps):
			theta_b = self.fit_(mini_batches, self.cycle_per_step)
			self.theta_history = np.hstack((self.theta_history, theta_b[0]))
			self.b_history = np.hstack((self.b_history, theta_b[1]))
		
		# -- Performing the last iterations -- #
		theta_b = self.fit_(mini_batches, self.remind_iter)
		self.theta_history = np.hstack((self.theta_history, theta_b[0]))
		self.b_history = np.hstack((self.b_history, theta_b[1]))
		
		return (self.theta, self.b)


	def _stochastic_fit_w_momentum_history_(self, x:np.ndarray, y:np.ndarray):
		"""
		...Docstring...
		"""
		for ii in range(self.steps):
			theta_b = self.fit_(x, y, start=ii * self.cycle_per_step, n_cycle=self.cycle_per_step)
			self.theta_history = np.hstack((self.theta_history, theta_b[0]))
			self.b_history = np.hstack((self.b_history, theta_b[1]))

		# -- Performing the last iterations -- #
		theta_b = self.fit_(x, y, start=ii * self.cycle_per_step, n_cycle=self.remind_iter)
		self.theta_history = np.hstack((self.theta_history, theta_b[0]))
		self.b_history = np.hstack((self.b_history, theta_b[1]))
		
		return (self.theta, self.b)


	def cost_history_(self, x:np.ndarray, y:np.ndarray):
		"""
		... Docstring ...
		"""
		if x.ndim != 2:
			str_err = "Incorrect dimension for x."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if x.shape[1] != self.theta_history.shape[0]:
			str_err = "Mismatching shape between x and theta."
			raise Exception(str_err)
		if x.shape[0] != y.shape[0]:
			str_err = "Mismatching shape between x and y."
			raise Exception(str_err)
		if y.shape[1] != 1:
			str_err = "Incorrect y shape: y must be (m, 1) shaped."
			raise Exception(str_err)
		epsilon = 1e-5
		m = x.shape[0]
		pred = self.predict_history_(x)
		cost = -(1/m) * (np.multiply(y, np.log(pred + epsilon)) + np.multiply(1 - y, np.log(1 - pred + epsilon)))
		cost = np.sum(cost, axis=0) + self.lambd * np.sum(np.square(self.theta))
		return cost


class MyLogisticMetrics():
	def accuracy_score_(y:np.ndarray, yhat:np.ndarray, pos_label=1):
		"""
		...Docstring...
		"""
		if yhat.ndim != 2:
			str_err = "Incorrect dimension for yhat."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if yhat.shape != y.shape:
			str_err = "Mismatching shape between yhat and y."
			raise Exception(str_err)
		tp_arr = (y == pos_label) & (yhat == pos_label)
		fp_arr = (y != pos_label) & (yhat == pos_label)
		tn_arr = (y != pos_label) & (yhat != pos_label)
		fn_arr = (y == pos_label) & (yhat != pos_label)
		tp = tp_arr.sum()
		fp = fp_arr.sum()
		tn = tn_arr.sum()
		fn = fn_arr.sum()
		if (tp == 0) & (fp == 0) & (tn == 0) & (fn == 0):
			accuracy = 0
		else:
			accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
		return round(accuracy, 4)


	def precision_score_(y:np.ndarray, yhat:np.ndarray, pos_label=1):
		"""
		...Docstring...
		"""
		if yhat.ndim != 2:
			str_err = "Incorrect dimension for yhat."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if yhat.shape != y.shape:
			str_err = "Mismatching shape between yhat and y."
			raise Exception(str_err)
		if pos_label not in np.unique(y):
			str_err = f"{pos_label} is not a possible value of y"
			raise Exception(str_err)
		tp_arr = (y == pos_label) & (yhat == pos_label)
		fp_arr = (y != pos_label) & (yhat == pos_label)
		tp = tp_arr.sum()
		fp = fp_arr.sum()
		precision = tp / (tp + fp + eps)
		return round(precision, 4)


	def recall_score_(y:np.ndarray, yhat:np.ndarray, pos_label=1):
		"""
		...Docstring...
		"""
		if yhat.ndim != 2:
			str_err = "Incorrect dimension for yhat."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if yhat.shape != y.shape:
			str_err = "Mismatching shape between yhat and y."
			raise Exception(str_err)
		if pos_label not in np.unique(y):
			str_err = f"{pos_label} is not a possible value of y"
			raise Exception(str_err)
		tp_arr = (y == pos_label) & (yhat == pos_label)
		fn_arr = (y == pos_label) & (yhat != pos_label)
		tp = tp_arr.sum()
		fn = fn_arr.sum()
		recall = tp / (tp + fn + eps)
		return round(recall, 4)


	def specificity_score_(y:np.ndarray, yhat:np.ndarray, pos_label=1):
		"""
		...Docstring...
		"""
		if yhat.ndim != 2:
			str_err = "Incorrect dimension for yhat."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if yhat.shape != y.shape:
			str_err = "Mismatching shape between yhat and y."
			raise Exception(str_err)
		tp_arr = (y == pos_label) & (yhat == pos_label)
		fp_arr = (y != pos_label) & (yhat == pos_label)
		tn_arr = (y != pos_label) & (yhat != pos_label)
		fn_arr = (y == pos_label) & (yhat != pos_label)

		tp = tp_arr.sum()
		fp = fp_arr.sum()
		tn = tn_arr.sum()
		fn = fn_arr.sum()
		specificity = tn / (tn + fp + eps)
		return round(specificity, 4)


	def f1_score_(y:np.ndarray, yhat:np.ndarray, pos_label=1):
		"""
		...Docstring...
		"""
		if yhat.ndim != 2:
			str_err = "Incorrect dimension for yhat."
			raise Exception(str_err)
		if y.ndim != 2:
			str_err = "Incorrect dimension for y."
			raise Exception(str_err)
		if yhat.shape != y.shape:
			str_err = "Mismatching shape between yhat and y."
			raise Exception(str_err)
		precision = MyLogisticMetrics.precision_score_(y, yhat, pos_label)
		recall = MyLogisticMetrics.recall_score_(y, yhat, pos_label)
		f1 = 2 * precision * recall / (precision + recall + eps)
		return round(f1, 4)


	def confusion_matrix_(y:np.ndarray, yhat:np.ndarray, labels=None, df_option=True):
		"""
		...Docstring...
		"""
		if labels is None:
			labels = np.unique(y).astype(object)
		confusion_matrix = pd.DataFrame(data=np.zeros((labels.shape[0], labels.shape[0])),
										index=labels,
										columns=labels)
		for index in labels:
			mask = y == index
			for col in labels:
				nb = np.sum(yhat[mask] == col)
				confusion_matrix[col][index] = nb
		if df_option == True:
			return confusion_matrix
		else:
			return confusion_matrix.values