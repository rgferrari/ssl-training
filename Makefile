# --- Adicionamos uma variável ALGO com um valor padrão ---
# Se você não especificar um algoritmo, ele usará 'ppo' por padrão.
# O '?=' significa 'defina esta variável APENAS se ela ainda não foi definida'.
ALGO ?= ppo

VENV := ssl-training-venv
PYTHON := python3.10
RSOCCER := rSoccer

.PHONY: all setup install clean train eval

all: setup install

setup:
	$(PYTHON) -m venv $(VENV)

install:
	# Clona apenas se o diretório não existir
	[ -d "$(RSOCCER)" ] || git clone https://github.com/robocin/rSoccer.git $(RSOCCER)
	$(VENV)/bin/pip install -e $(RSOCCER)
	$(VENV)/bin/pip install -e .

clean:
	rm -rf $(VENV)
	rm -rf $(RSOCCER)

# --- O target 'train' agora usa a variável ALGO para encontrar o JSON ---
train:
	$(PYTHON) main.py --train --json $(ALGO).json

# --- O target 'eval' também foi atualizado para consistência ---
eval:
	$(PYTHON) main.py --json $(ALGO).json
