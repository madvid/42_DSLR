import numpy as np

# =========================================================================== #
#                        | Definition des constantes|                         #
# =========================================================================== #
END = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
BLACK = '\033[1;30m'
RED = '\033[1;31m'
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[1;34m'
VIOLET = '\033[1;35m'
CYAN = '\033[1;36m'
WHITE = '\033[1;37m'

default_train_file = "datasets/dataset_train.csv"
default_predict_file = "datasets/dataset_test.csv"

# Les series d'intérêt dans le dataset de train ou de test sont les clefs du dct
# suivant. Nous fixons le type de ces series
dct_types = {"Index" : np.int32, "Hogwarts House" : str, "Fisrt Name" : str,
			 "Last Name" : str, "Birthday" : str, "Best Hand" : str,
			 "Arithmancy" : np.float32, "Astronomy" : np.float32,
			 "Herbology" : np.float32, "Defense Against the Dark Arts" : np.float32,
			 "Divination" : np.float32, "Muggle Studies" : np.float32,
			 "Ancient Runes" : np.float32, "History of Magic" : np.float32,
			 "Transfiguration" : np.float32, "Potions" : np.float32,
			 "Care of Magical Creatures" : np.float32, "Charms" : np.float32,
			 "Flying" : np.float32}

other_cols = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
numerical_cols = ["Arithmancy", "Astronomy", "Herbology",
				  "Defense Against the Dark Arts", "Divination",
				  "Muggle Studies", "Ancient Runes", "History of Magic",
				  "Transfiguration", "Potions", "Care of Magical Creatures",
				  "Charms", "Flying"]

expected_col = ["Index", "Hogwarts House", "First Name", "Last Name",
				"Birthday", "Best Hand", "Arithmancy", "Astronomy",
				"Herbology", "Defense Against the Dark Arts", "Divination",
				"Muggle Studies", "Ancient Runes", "History of Magic",
				"Transfiguration", "Potions", "Care of Magical Creatures",
				"Charms", "Flying"]

# -- values to predict -- #
target = "Hogwarts House"
lst_classes = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
nb_classes = len(lst_classes)

# -- features used in the models -- #
#lst_features = ["Arithmancy", "Herbology", "Defense Against the Dark Arts", "Divination",
#					"Muggle Studies", "Ancient Runes", "History of Magic",
#					"Transfiguration", "Potions", "Care of Magical Creatures",
#					"Charms", "Flying"]
lst_features = ["Defense Against the Dark Arts", "Herbology", "Divination",
					"Ancient Runes", "History of Magic"]

nb_features = len(lst_features)

# -- training / dev set ratio -- #
split_ratio = 0.8