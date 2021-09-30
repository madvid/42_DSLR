from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import matplotlib.image as pltimg

from utils import expected_col, RED, END

if __name__ == "__main__":
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
	
	try:
		df = read_csv("datasets/dataset_train.csv", dtype=dct_types)
	except:
		print(RED + "At least one column is missing or is not of excpected dtype." + END)
		sys.exit()

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

	# --- Editing some column names on the dataframe --- #
	df.rename(columns={"Defense Against the Dark Arts":"Def. vs DA"}, inplace=True)
	df.rename(columns={"Care of Magical Creatures":"Care of\nMagical Creatures"}, inplace=True)
	df.rename(columns={"History of Magic":"History\nof Magic"}, inplace=True)
	
	df.dropna(inplace=True)
	df = df.sample(n=int(0.25 * df.shape[0])) # sampling because the dataset is big and take time to plot

	# --- Plotting part -- #
	dct_palet = {"Hufflepuff":"dodgerblue", "Gryffindor":"red", "Slytherin":"green", "Ravenclaw":"goldenrod"}

	sns_pairplot = sns.pairplot(data=df,
								hue="Hogwarts House",
								palette=dct_palet,
								plot_kws=dict(s=10, alpha=0.6),
								height=0.95, aspect=1.4, corner=True)
	sns_pairplot._legend.set_bbox_to_anchor((0.6, 0.6))
	plt.show()
