from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from utils import expected_col, RED, END

if __name__ == "__main__":
	df = read_csv("datasets/dataset_train.csv")
	name_cols = df.columns.values
	
	# --- Basic verification on the dataset -- #
	if not (all([name in expected_col for name in name_cols]) \
		and ([col in name_cols for col in expected_col])):
		print(RED + "Dataset does not contain exactly the expected columns." + END)
		sys.exit()
	
	if df.shape[1] != 19:
		print(RED + "Dataset is expected to have 19 columns." + END)
		sys.exit()

	if df.shape[0] == 0:
		print(RED + "Dataset seems to be empty." + END)
		sys.exit()
	
	# --- Plotting part -- #
	c_palet = ["goldenrod", "green", "red", "dodgerblue"]
	sns.scatterplot(data=df, x="Astronomy", y="Defense Against the Dark Arts", hue="Hogwarts House", palette=c_palet)
	plt.show()
