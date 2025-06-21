# Makefile for MLOps project

.PHONY: help setup train serve test lint format clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup    - Install dependencies"
	@echo "  make train    - Train the model"
	@echo "  make serve    - Run the API server"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run linting"
	@echo "  make format   - Format code"
	@echo "  make clean    - Clean generated files"
	@echo "  make docker   - Build and run with Docker"

setup:
	pip install -r requirements.txt

train: setup
	mkdir -p models data logs
	python src/train.py

serve:
	python src/predict.py

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src tests
	isort --check-only src tests
	black --check src tests

format:
	isort src tests
	black src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

docker:
	docker-compose up --build