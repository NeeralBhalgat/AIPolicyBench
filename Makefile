.PHONY: run-white run-green build test clean

# Variables
PYTHON := python
DOCKER_IMAGE := aipolicybench-white-agent
PORT := 9002
export PYTHONPATH := .

# Run White Agent (Locally)
run-white:
	$(PYTHON) -m src.white_agent.agent

# Run Green Agent (Evaluator)
# Assumes green agent is still in the old location or moved later
run-green:
	$(PYTHON) green_agent/agent.py

# Build Docker Image
build:
	docker build -t $(DOCKER_IMAGE) .

# Run Docker Container
run-docker:
	docker run -p $(PORT):$(PORT) --env-file .env $(DOCKER_IMAGE)

# Run Tests
test:
	$(PYTHON) -m pytest tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
