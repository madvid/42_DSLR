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
	
	# -- Minor formatting of the datafram -- #
	df["Birthday"] = df["Birthday"].astype("datetime64")



	# --- Plotting part -- #
	c_palet = ["goldenrod", "green", "red", "dodgerblue"]
	fig, axes = plt.subplots(3, 5, figsize=(20, 12))

	sns.histplot(data=df, x="Birthday", hue="Hogwarts House", stat="probability", binwidth=90, palette=c_palet, multiple="dodge", legend=None, ax=axes[0,0])
	sns.histplot(data=df, x="Best Hand", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[0,1])
	sns.histplot(data=df, x="Arithmancy", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[0,2])
	sns.histplot(data=df, x="Astronomy", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[0,3])
	sns.histplot(data=df, x="Herbology", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[0,4])
	sns.histplot(data=df, x="Defense Against the Dark Arts", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[1,0])
	sns.histplot(data=df, x="Divination", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[1,1])
	sns.histplot(data=df, x="Muggle Studies", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[1,2])
	sns.histplot(data=df, x="Ancient Runes", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[1,3])
	sns.histplot(data=df, x="History of Magic", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[1,4])
	sns.histplot(data=df, x="Transfiguration", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[2,0])
	sns.histplot(data=df, x="Potions", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[2,1])
	sns.histplot(data=df, x="Care of Magical Creatures", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[2,2])
	sns.histplot(data=df, x="Charms", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[2,3])
	sns.histplot(data=df, x="Flying", hue="Hogwarts House", stat="probability", palette=c_palet, multiple="dodge", legend=None, ax=axes[2,4])

	axes[0,1].set_ylabel("")
	axes[0,2].set_ylabel("")
	axes[0,3].set_ylabel("")
	axes[0,4].set_ylabel("")
	axes[1,1].set_ylabel("")
	axes[1,2].set_ylabel("")
	axes[1,3].set_ylabel("")
	axes[1,4].set_ylabel("")
	axes[2,1].set_ylabel("")
	axes[2,2].set_ylabel("")
	axes[2,3].set_ylabel("")
	axes[2,4].set_ylabel("")
	axes[2,2].legend(loc=5, labels=["Hufflepuff", "Gryffindor", "Slytherin", "Ravenclaw"], bbox_to_anchor=(4.03,1.5), fontsize="large")
	plt.show()