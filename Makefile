VENV := ssl-training-venv
PYTHON := python3.10
RSOCCER := rSoccer

.PHONY: all setup install clean train eval

all: setup install

setup:
	$(PYTHON) -m venv $(VENV)

install:
	git clone https://github.com/robocin/rSoccer.git $(RSOCCER)
	$(VENV)/bin/pip install -e $(RSOCCER)
	$(VENV)/bin/pip install -e .

clean:
	rm -rf $(VENV)
	rm -rf $(RSOCCER)

train:
	$(PYTHON) main.py --train --json hyperparameters.json

eval:
	$(PYTHON) main.py --json hyperparameters.json