.PHONY: help install test train tune walkforward report clean

help:
	@echo "VIX ETP RL Module - Available Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make train        - Train PPO agent"
	@echo "  make tune         - Run hyperparameter tuning"
	@echo "  make walkforward  - Run walk-forward validation"
	@echo "  make report       - Generate performance report"
	@echo "  make clean        - Clean output files"
	@echo ""

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

train:
	python scripts/cli.py train --agent ppo --timesteps 100000

train-sac:
	python scripts/cli.py train --agent sac --timesteps 100000

tune:
	python scripts/cli.py tune --trials 50 --timesteps 50000

walkforward:
	python scripts/cli.py walkforward --agent ppo --timesteps 50000

report:
	python scripts/cli.py report

eval:
	python scripts/cli.py eval --model outputs/vix_etp_ppo_model.zip --agent ppo

clean:
	rm -rf outputs/*
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

quick-test:
	pytest tests/ -q --tb=no

coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
