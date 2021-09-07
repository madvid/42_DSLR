# =========================================================================== #
#                       |Importation des lib/packages|                        #
# =========================================================================== #
from utils import print_train_usage
import sys

# =========================================================================== #
#                        | Definition des constantes|                         #
# =========================================================================== #
lst_possible_args = ["--graphic=console",
					 "--graphic=static",
					 "--graphic=dynamic",
					 "--method=gradient-descent",
					 "--method=sotchastic-gradient-descent",
					 "--method=minibatch"]


# =========================================================================== #
#                        | Definition des functions|                          #
# =========================================================================== #

def parser(argv:list):
	"""
	... Docstring ...
	"""
	b_visu = b_static = b_console = b_gd = \
		b_sgd = b_minibatch = b_method = False

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
		elif (arg == "--method=minibatch") and (b_method == False):
			b_method = True
			b_minibatch = True
			method = arg.split('=')[1]
		else:
			ag_dataset = arg.split('=')
			if len(ag_dataset) == 2:
				if ag_dataset[0] == "--dataset":
					datapath = ag_dataset[1] 
				else:
					str_expt = "Dataset argument is incorrect."
					print(str_expt)
					sys.exit()
			elif arg not in lst_possible_args:
				str_expt = "Invalid argument."
				print(str_expt)
				sys.exit()
			else:
				str_expt = "Method or graphic argument cannot be define more than once."
				print(str_expt)
				sys.exit()

	if not b_method:
		b_method = b_gd = True
	if not b_visu:
		b_visu = b_console = True

	return datapath, b_visu, b_static, b_console, b_gd, b_sgd, b_minibatch, b_method