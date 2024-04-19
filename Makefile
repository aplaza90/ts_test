
run:
	@echo "Executing the pipeline..."
	python3 src/pipeline.py

setup:
	@echo "Setting up virtual environment and installing dependencies..."
	python3.7 -m venv .venv
	source .venv/bin/activate; \
	pip install -r requirements.txt