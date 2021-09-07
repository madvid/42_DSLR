### PROJECT ###
NAME = dslr

### VIRTUAL ENVIRONMENT ###
V_NAME = v_dslr

### OS DEPENDING RULES ###
OS = $(shell uname)

### COLORS ###
NOC = \033[0m
BOLD = \033[1m
UNDERLINE = \033[4m
BLACK = \033[1;30m
RED = \033[1;31m
GREEN = \033[1;32m
YELLOW = \033[1;33m
BLUE = \033[1;34m
VIOLET = \033[1;35m
CYAN = \033[1;36m
WHITE = \033[1;37m

### RULES ###

all: install

install:
	@echo "$(CYAN)Installing python3.9 and Generating virtual environment$(NOC): $(GREEN)$(V_NAME)$(NOC)"
	@if [ $(OS) =  "Linux" ]; then \
		sudo apt update -y; \
		sudo apt install python3.9 python3-pip python3.9-tk; \
		python3.9 -m venv $(V_NAME); \
		. $(V_NAME)/bin/activate && python -m pip install --upgrade pip && pip install -U -r requirements.txt; \
	fi
	@if [ $(OS) = "Darwin" ]; then \
		brew install python@3.9; \
		python3.9 -m venv $(V_NAME); \
		source $(V_NAME)/bin/activate && python -m pip install --upgrade pip && pip install -U -r requirements.txt; \
	fi
	@echo "$(GREEN)Virtual environment fully sets up.$(NOC)";
	@echo "$(YELLOW)Run \"source $(V_NAME)/bin/activate\" to activate the environment$(NOC)";

clean:
	@echo "$(CYAN)Supressing pycache$(NOC)"
	@rm -rf __pycache__
	@echo "$(RED)pycache destroyed$(NOC)"
	@echo "$(CYAN)Supressing models.json$(NOC)"
	@rm -f models.json
	@echo "$(RED)models.json destroyed$(NOC)"

fclean: clean
	@echo "$(CYAN)Supressing venv repository:$(NOC)"
	@rm -rf $(V_NAME)
	@echo "$(RED)venv destroyed$(NOC)"

re: fclean all

.PHONY: install, clean, fclean, re
