# =========================================================================== #
#                       |Importation des lib/packages|                        #
# =========================================================================== #
from utils import print_train_usage, print_predict_usage
import sys
from os.path import islink
from constants import RED, END
# =========================================================================== #
#                        | Definition des constantes|                         #
# =========================================================================== #
lst_possible_args = ["--graphic=console",
					 "--graphic=static",
					 "--method=gradient-descent",
					 "--method=sotchastic-gradient-descent",
					 "--method=stochastic-gradient-descent+momentum",
					 "--method=minibatch"]

forbiddens = ["/dev/random", "/dev/zero", "/dev/null"]

# =========================================================================== #
#                        | Definition des functions|                          #
# =========================================================================== #
def parser(argv:list):
	"""
	...Docstring...
	"""
	b_visu = b_static = b_console = b_gd = \
		b_sgd = b_sgd_moment = b_minibatch = b_method = b_data = False

	if len(argv) == 1 and (argv[0] in ["-h", "--help", "--usage"]):
		print_train_usage()
		sys.exit()
	for arg in argv:
		if (arg == "--graphic=console") and (b_visu == False):
			b_visu = True
			b_console = True
		elif (arg == "--graphic=static") and (b_visu == False):
			b_visu = True
			b_static = True
		elif (arg == "--method=gradient-descent") and (b_method == False):
			b_method = True
			b_gd = True
			method = arg.split('=')[1]
		elif (arg == "--method=stochastic-gradient-descent") and (b_method == False):
			b_method = True
			b_sgd = True
			method = arg.split('=')[1]
		elif (arg == "--method=stochastic-gradient-descent+momentum") and (b_method == False):
			b_method = True
			b_sgd_moment = True
			method = arg.split('=')[1]
		elif (arg == "--method=minibatch") and (b_method == False):
			b_method = True
			b_minibatch = True
			method = arg.split('=')[1]
		else:
			ag = arg.split('=')
			if (len(ag) == 2):
				if (ag[0] == "--dataset") and (b_data == False):
					b_data = True
					datapath = ag[1] 
				elif arg in lst_possible_args:
					str_expt = "Multiple definition of method, graphic or data argument."
					print(str_expt)
					sys.exit()
				else:
					str_expt = "Invalid argument or invalid value for method/graphic."
					print(str_expt)
					sys.exit()
			else:
				str_expt = "Invalid argument."
				print(str_expt)
				sys.exit()

	if not b_method:
		b_method = b_gd = True
		print(f"Method argument is not specified, " + \
			"value will be set to 'gradient descent'.")
	if not b_visu:
		b_visu = b_console = True
		print(f"Visulization argument is not specified, " + \
			"value will be set to 'console'.")
	if not b_data:
		datapath = "datasets/dataset_train.csv"
		print("Dataset argument is not specified, " + \
			f"value will be set to '{datapath}'.")
	
	if islink(datapath) or (datapath in forbiddens) \
		or (datapath.split('..')[-1] in forbiddens):
		str_expt = "No links authorized or used of /dev/[random | zero | null]."
		print(RED + str_expt + END)
		sys.exit()
	
	return datapath, b_visu, b_static, b_console, b_gd, b_sgd, b_sgd_moment, b_minibatch, b_method


def parser_predict(argv:list):
	"""
	...Docstring...
	"""
	b_data = b_model = False

	if len(argv) == 1 and (argv[0] in ["-h", "--help", "--usage"]):
		print_predict_usage()
		sys.exit()
	for arg in argv:
		ag = arg.split('=')
		if (len(ag) == 2):
			if (ag[0] == "--dataset") and (b_data == False):
				if ag[1] == "":
					str_expt = "There is no value for dataset arg..."
					print(str_expt)
					sys.exit()
				b_data = True
				datapath = ag[1] 
			elif (ag[0] == "--dataset") and (b_data == True):
				str_expt = "Multiple dataset argument."
				print(str_expt)
				sys.exit()
			elif (ag[0] == "--model") and (b_model == False):
				if ag[1] == "":
					str_expt = "There is no value for model arg..."
					print(str_expt)
					sys.exit()
				b_model = True
				modelpath = ag[1]
			elif (ag[0] == "--model") and (b_model == True):
				str_expt = "Multiple model argument."
				print(str_expt)
				sys.exit()
			else:
				str_expt = "At least one argument is incorrect."
				print(str_expt)
				sys.exit()
		else:
			str_expt = "At least one argument is incorrect."
			print(str_expt)
			sys.exit()
	if not b_data:
		datapath = "datasets/dataset_test.csv"
		print(f"Dataset is not specified, value will be set to {datapath}.")
	if not b_model:
		modelpath = "model.json"
		print(f"Model is not specified, value will be set to {modelpath}.")
	
	if islink(datapath) or islink(modelpath) \
		or (datapath in forbiddens) or (modelpath in forbiddens) \
			or (datapath.split('..')[-1] in forbiddens):
		str_expt = "No links authorized or used of /dev/[random | zero | null]."
		print(RED + str_expt + END)
		sys.exit()
	
	return datapath, modelpath