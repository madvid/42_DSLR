import numpy as np
import pandas as pd
from utils import *
import tinystatistician as tstat
from sys import argv, exit

# --------------------------------------------------------------------------- #
#           program arguments and verbose variable definitions                #
# --------------------------------------------------------------------------- #
data_file = ["datasets/dataset_test.csv", "datasets/dataset_train.csv"]
invalid_arg = "First parameter is not a dataset or any flags for help/usage."
invalid_csv = "Dataset is not valid."
lst_usage = ["-h", "--help", "--usage"]

def usage_display():
	""" Display the usage of the program describe.py
	"""
	str_usage = GREEN + "Usage:\n" + END
	str_usage += f"  python describe.py {YELLOW}dataset.csv{END}"
	print(str_usage)

# --------------------------------------------------------------------------- #
# __________________________________ MAIN ___________________________________ #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
	b_all = False

	argv = argv[1:]
	if len(argv) == 0:
		print(RED + "No argument has been given: python describe.py -h/--help/--usage" + END)
		exit()
	if argv[0] in lst_usage:
		usage_display()
		exit()
	elif (len(argv) == 1) and (argv[0] == "--dataset=all"):
		df = open_read_file("all")
	elif (len(argv) == 1) and (argv[0] in data_file):
		df = open_read_file(argv[0].split("/")[1])
	else:
		print(invalid_arg)
		exit()
	
	if df is None:
		exit()
	df_shape = df.values.shape
	col_names = df.columns.values
	df_count = pd.DataFrame(np.full((1, df_shape[1]), df_shape[0]) , columns=col_names)
	df_mean = pd.DataFrame(tstat.mean(df.copy(deep=True).values), columns=col_names)
	df_std = pd.DataFrame(tstat.std(df.copy(deep=True).values, df_mean.values[0].reshape(-1,)), columns=col_names)
	df_min = pd.DataFrame(tstat.min(df.copy(deep=True).values), columns=col_names)
	df_max = pd.DataFrame(tstat.max(df.copy(deep=True).values), columns=col_names)
	df_percentile25 = pd.DataFrame(tstat.percentile(df.copy(deep=True).values, 25), columns=col_names)
	df_percentile50 = pd.DataFrame(tstat.percentile(df.copy(deep=True).values, 50), columns=col_names)
	df_percentile75 = pd.DataFrame(tstat.percentile(df.copy(deep=True).values, 75), columns=col_names)

	df_count.index = pd.Index(["count"])
	df_mean.index = pd.Index(["mean"])
	df_std.index = pd.Index(["std"])
	df_min.index = pd.Index(["min"])
	df_max.index = pd.Index(["max"])
	df_percentile25.index = pd.Index(["25%"])
	df_percentile50.index = pd.Index(["50%"])
	df_percentile75.index = pd.Index(["75%"])
	df_describe = pd.concat([df_count,
							 df_mean,
							 df_std,
							 df_min,
							 df_percentile25,
							 df_percentile50,
							 df_percentile75,
							 df_max])
	print(df_describe)
	